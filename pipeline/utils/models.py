"""
Model Orchestrator for Browser-Use Pipeline

This module intelligently selects the best AI model for different tasks in the pipeline.
It discovers available models, categorizes them by capability, and caches the results.

Key Concepts:
- Different tasks (planning, coding, testing) need different model strengths
- Models are auto-discovered from your environment or fall back to known good models
- Results are cached to avoid repeated discovery overhead
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import ollama

logger = logging.getLogger(__name__)


class TaskRole(str, Enum):
    """Specific roles assigned to AI models within the processing pipeline.

    Used by the `ModelOrchestrator` to match tasks with the most appropriate
    model capabilities (e.g., using 'Pro' models for planning).

    Attributes:
        PLANNER: High-level strategic thinking and task decomposition.
        ENGINEER: Code generation and implementation tasks.
        TESTER: Quality assurance and validation checks.
        REVIEWER: Code review and optimization suggestions.
        THINKER: Deep reasoning and complex problem-solving.
    """
    PLANNER = "planner"
    ENGINEER = "engineer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    THINKER = "thinker"


class ModelOrchestrator:
    """Intelligent model discovery and role-based assignment manager.

    Maintains a registry of available LLM models (local via Ollama or remote)
    and maps them to specific processing roles based on their known or
    discovered capabilities. Features an aging cache to minimize discovery
    latency.
    """
    
    def __init__(self) -> None:
        """Initialize the orchestrator with configuration from environment."""
        self._cache_dir = Path(
            os.environ.get("PIPELINE_CACHE_DIR", "~/.cache/browser-use")
        ).expanduser()
        
        self._cache_file = self._cache_dir / "model_assignments.json"
        self._cache_ttl_seconds = int(os.environ.get("PIPELINE_CACHE_TTL", 86400))  # 24 hours
        self._default_model = os.environ.get("PIPELINE_DEFAULT_MODEL", "gemini-2.0-flash")
        
        # Configure Gemini
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        
        self._model_assignments: Optional[dict[str, list[str]]] = None
    
    def get_best_model_for_task(self, role: TaskRole | str) -> str:
        """
        Get the most suitable model for a specific task.
        
        Args:
            role: The task role (e.g., TaskRole.PLANNER or "planner")
            
        Returns:
            Model name suitable for the task (e.g., "gemini-2.0-flash")
            
        Example:
            >>> orchestrator = ModelOrchestrator()
            >>> model = orchestrator.get_best_model_for_task(TaskRole.ENGINEER)
            >>> print(model)  # "gemini-2.0-flash"
        """
        assignments = self._get_model_assignments()
        
        if not assignments:
            logger.warning(f"No model assignments found, using default: {self._default_model}")
            return self._default_model
        
        # Normalize role to uppercase string
        role_key = role.value.upper() if isinstance(role, TaskRole) else role.upper()
        
        # Get models assigned to this role
        role_models = assignments.get(role_key, [])
        
        if not role_models:
            logger.warning(f"No models found for role {role_key}, using default")
            return self._default_model
        
        # Return the best model (first in list, already sorted by capability)
        best_model = self._clean_model_name(role_models[0])
        logger.debug(f"Selected model '{best_model}' for role '{role_key}'")
        return best_model
    
    def _get_model_assignments(self) -> dict[str, list[str]]:
        """
        Load model-to-role assignments from cache or discover fresh.
        
        Returns:
            Dictionary mapping role names to lists of suitable models
        """
        if self._model_assignments is not None:
            return self._model_assignments
        
        # Try to load from cache first
        if self._is_cache_valid():
            cached = self._load_from_cache()
            if cached:
                self._model_assignments = cached
                return cached
        
        # Cache miss or invalid - discover models
        logger.info("Discovering available models and assigning to roles...")
        fresh_assignments = self._discover_and_assign_models()
        
        if fresh_assignments:
            self._save_to_cache(fresh_assignments)
            self._model_assignments = fresh_assignments
        
        return fresh_assignments or {}
    
    def _is_cache_valid(self) -> bool:
        """Check if the cache file exists and is still fresh."""
        if not self._cache_file.exists():
            return False
        
        age_seconds = time.time() - self._cache_file.stat().st_mtime
        return age_seconds < self._cache_ttl_seconds
    
    def _load_from_cache(self) -> Optional[dict[str, list[str]]]:
        """Load model assignments from the cache file."""
        try:
            with open(self._cache_file, "r") as f:
                data = json.load(f)
                logger.debug(f"Loaded model assignments from cache: {self._cache_file}")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, assignments: dict[str, list[str]]) -> None:
        """Save model assignments to the cache file."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(assignments, f, indent=2)
            logger.debug(f"Saved model assignments to cache: {self._cache_file}")
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _discover_and_assign_models(self) -> dict[str, list[str]]:
        """
        Discover available models and assign them to roles based on capabilities.
        
        Strategy:
        - Gemini Pro/Thinking models → Planning and deep reasoning
        - Gemini Flash models → Fast engineering/coding tasks
        - Other providers (Claude, GPT) → Backup options
        
        Returns:
            Dictionary mapping role names to prioritized model lists
        """
        available_models = self._discover_available_models()
        
        if not available_models:
            logger.warning("No models discovered")
            return {}
        
        # Separate Gemini models from others
        gemini_models = [m for m in available_models if "google/gemini" in m.lower()]
        other_models = [m for m in available_models if "google/gemini" not in m.lower()]
        
        # Assign models to roles based on capabilities
        assignments = {
            "PLANNER": self._find_planning_models(gemini_models, other_models),
            "ENGINEER": self._find_engineering_models(gemini_models, other_models),
            "TESTER": self._find_testing_models(gemini_models, other_models),
            "THINKER": self._find_reasoning_models(gemini_models, other_models),
        }
        
        # Ensure all roles have at least one model
        for role, models in assignments.items():
            if not models:
                logger.warning(f"No specific models for {role}, using all available")
                assignments[role] = available_models
        
        return assignments
    
    def _discover_available_models(self) -> list[str]:
        """
        Discover models using official Gemini and Ollama SDKs.
        
        Returns:
            List of available model identifiers with provider prefix.
        """
        models = []
        
        # 1. Discover Gemini Models
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    name = m.name
                    if name.startswith("models/"):
                        name = name[7:]
                    models.append(f"google/{name}")
        except Exception as e:
            logger.debug(f"Gemini discovery skipped (check API key): {e}")
        
        # 2. Discover Ollama Models
        try:
            ollama_models = ollama.list()
            # Handle both old and new ollama-python response formats
            model_list = ollama_models.get('models', []) if isinstance(ollama_models, dict) else ollama_models
            for m in model_list:
                name = m.get('name') if isinstance(m, dict) else getattr(m, 'model', None)
                if name:
                    models.append(f"ollama/{name}")
        except Exception as e:
            logger.debug(f"Ollama discovery skipped (is server running?): {e}")
            
        return models
    
    def _find_planning_models(self, gemini: list[str], others: list[str]) -> list[str]:
        """Find models best suited for high-level planning tasks."""
        # Prefer Gemini Pro and Thinking models
        candidates = self._filter_and_sort(
            gemini, 
            must_include=["pro", "thinking"]
        )
        
        # Add capable models from other providers
        if not candidates:
            candidates = [
                m for m in others 
                if any(keyword in m.lower() for keyword in ["opus", "o1", "pro"])
            ]
        
        return candidates
    
    def _find_engineering_models(self, gemini: list[str], others: list[str]) -> list[str]:
        """Find models best suited for code generation."""
        # Prefer fast Gemini Flash models
        candidates = self._filter_and_sort(
            gemini,
            must_include=["flash"],
            must_exclude=["pro"]  # Flash-only, not Flash-Pro
        )
        
        # Backup to other fast models
        if not candidates:
            candidates = [
                m for m in others
                if any(keyword in m.lower() for keyword in ["sonnet", "flash", "coder"])
            ]
        
        return candidates
    
    def _find_testing_models(self, gemini: list[str], others: list[str]) -> list[str]:
        """Find models best suited for testing and validation."""
        # Prefer lighter, faster models
        candidates = self._filter_and_sort(
            gemini,
            must_include=["lite", "flash"]
        )
        
        if not candidates:
            candidates = [
                m for m in others
                if any(keyword in m.lower() for keyword in ["haiku", "flash", "mini"])
            ]
        
        return candidates
    
    def _find_reasoning_models(self, gemini: list[str], others: list[str]) -> list[str]:
        """Find models best suited for deep reasoning."""
        # Prefer models with explicit reasoning capabilities
        candidates = self._filter_and_sort(gemini, must_include=["thinking"])
        
        # Fall back to planning models if no thinking models available
        return candidates if candidates else self._find_planning_models(gemini, others)
    
    def _filter_and_sort(
        self, 
        models: list[str], 
        must_include: list[str],
        must_exclude: Optional[list[str]] = None
    ) -> list[str]:
        """
        Filter models by keywords and sort by version number.
        
        Args:
            models: List of model names to filter
            must_include: Keywords that must be present (OR logic)
            must_exclude: Keywords that must not be present
            
        Returns:
            Filtered and sorted model list (newest versions first)
        """
        must_exclude = must_exclude or []
        
        # Filter by inclusion/exclusion criteria
        filtered = [
            model for model in models
            if any(keyword in model.lower() for keyword in must_include)
            and not any(keyword in model.lower() for keyword in must_exclude)
        ]
        
        # Sort by version (newest first)
        return sorted(filtered, key=self._extract_version, reverse=True)
    
    @staticmethod
    def _extract_version(model_name: str) -> float:
        """
        Extract version number from model name for sorting.
        
        Examples:
            "gemini-2.0-flash" → 2.0
            "gpt-4o" → 4.0
            "claude-3-opus" → 3.0
            
        Returns:
            Version as float, or 0.0 if no version found
        """
        # Look for patterns like "2.0", "3.5", "4", etc.
        match = re.search(r"(\d+(?:\.\d+)?)", model_name)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.0
    
    @staticmethod
    def _clean_model_name(model: str) -> str:
        """
        Remove provider prefix from model name.
        
        Examples:
            "google/gemini-2.0-flash" → "gemini-2.0-flash"
            "ollama/llama3" → "llama3"
            "gemini-2.0-flash" → "gemini-2.0-flash" (unchanged)
        """
        if "/" not in model:
            return model
        
        parts = model.split("/", 1)
        if len(parts) == 2 and parts[0] in ["google", "openai", "anthropic", "ollama"]:
            return parts[1]
        
        return model
    
    def clear_cache(self) -> None:
        """Clear the cached model assignments, forcing fresh discovery next time."""
        if self._cache_file.exists():
            self._cache_file.unlink()
            logger.info("Cache cleared")
        self._model_assignments = None


# Convenience functions for backward compatibility
def get_model_for_role(role: TaskRole | str) -> str:
    """
    Get the best model for a given role (convenience function).
    
    This creates a new orchestrator each time. For better performance,
    create a ModelOrchestrator instance and reuse it.
    
    Args:
        role: The task role
        
    Returns:
        Model name suitable for the task
    """
    orchestrator = ModelOrchestrator()
    return orchestrator.get_best_model_for_task(role)


# Maintain backward compatibility with old enum name
AgentRole = TaskRole