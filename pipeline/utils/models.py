from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Dict, List

# Version: 1.2
CACHE_FILE = os.path.expanduser("~/.cache/ralph/model_segments.json")
CACHE_TTL = 86400  # 24 hours
DEFAULT_MODEL = "google/gemini-3-flash"


def extract_version(name: str) -> float:
    """Extract numeric version from model name."""
    match = re.findall(r"(\d+\.?\d*)", name)
    if match:
        try:
            val = match[0]
            if len(match) > 1 and "." not in val:
                val = f"{match[0]}.{match[1]}"
            return float(val)
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def get_available_models() -> List[str]:
    """Get list of available models from opencode or fallback list."""
    try:
        result = subprocess.run(
            ["opencode", "models"], capture_output=True, text=True, check=True
        )
        return [line.strip() for line in result.stdout.splitlines() if "/" in line]
    except (subprocess.SubprocessError, FileNotFoundError):
        # Fallback to known frontier models as of Jan 2026
        return [
            "google/gemini-3-pro",
            DEFAULT_MODEL,
            "google/gemini-3-thinking",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "openai/gpt-4o",
            "openai/o1-preview",
        ]


def _filter_models(
    models: List[str], includes: List[str], excludes: List[str] = None
) -> List[str]:
    """Filter models by inclusion/exclusion keywords and sort by version."""
    if excludes is None:
        excludes = []
    filtered = [
        m
        for m in models
        if any(k in m.lower() for k in includes)
        and not any(x in m.lower() for x in excludes)
    ]
    return sorted(filtered, key=extract_version, reverse=True)


def discover_and_segment() -> Dict[str, List[str]]:
    """Segment available models into functional roles."""
    available = get_available_models()
    if not available:
        return {}

    segments = {"PLANNER": [], "ENGINEER": [], "TESTER": [], "THINKER": []}
    gemini = [m for m in available if "google/gemini" in m.lower()]
    others = [m for m in available if "google/gemini" not in m.lower()]

    segments["PLANNER"] = _filter_models(gemini, ["pro", "thinking"])
    segments["ENGINEER"] = _filter_models(gemini, ["flash"], ["pro"])
    segments["TESTER"] = _filter_models(gemini, ["lite", "flash"])

    for role in segments:
        if not segments[role]:
            if role in ["PLANNER", "THINKER"]:
                segments[role] = [
                    m for m in others if any(k in m for k in ["opus", "o1", "pro"])
                ]
            else:
                segments[role] = [
                    m for m in others if any(k in m for k in ["sonnet", "flash", "coder"])
                ]
            if not segments[role]:
                segments[role] = available

    thinking = _filter_models(gemini, ["thinking"])
    segments["THINKER"] = thinking if thinking else segments["PLANNER"]
    return segments


def get_model_for_role(role: str) -> str:
    """Retrieve the best model for a given role, using cache if valid."""
    role = role.upper()
    current_time = time.time()

    use_cache = False
    if os.path.exists(CACHE_FILE):
        if (current_time - os.path.getmtime(CACHE_FILE)) < CACHE_TTL:
            use_cache = True

    segments = None
    if use_cache:
        try:
            with open(CACHE_FILE, "r") as f:
                segments = json.load(f)
        except (json.JSONDecodeError, IOError):
            segments = discover_and_segment()
    else:
        segments = discover_and_segment()
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(segments, f)
        except IOError:
            pass

    if not segments:
        return DEFAULT_MODEL

    role_models = segments.get(role, [DEFAULT_MODEL])
    if role_models:
        return role_models[0]

    return DEFAULT_MODEL