from __future__ import annotations

import json
import os
import re
import subprocess
import time

# Version: 1.2
CACHE_FILE = os.path.expanduser("~/.cache/ralph/model_segments.json")
CACHE_TTL = 86400  # 24 hours
DEFAULT_MODEL = "gemini-2.0-flash"


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


def get_available_models() -> list[str]:
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
    models: list[str], includes: list[str], excludes: list[str] | None = None
) -> list[str]:
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


def discover_and_segment() -> dict[str, list[str]]:
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


def _load_model_segments() -> dict[str, list[str]]:
    """Load model segments from cache or discover and save to cache."""
    current_time = time.time()
    if os.path.exists(CACHE_FILE) and (current_time - os.path.getmtime(CACHE_FILE)) < CACHE_TTL:
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    segments = discover_and_segment()
    if segments:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(segments, f)
        except IOError:
            pass
    return segments


def _format_model_name(model: str) -> str:
    """Strip provider prefix if present (e.g., 'google/gemini-1.5-flash' -> 'gemini-1.5-flash')."""
    if "/" in model:
        parts = model.split("/")
        if len(parts) > 1 and parts[0] in ["google", "openai", "anthropic"]:
            return "/".join(parts[1:])
    return model


def get_model_for_role(role: str) -> str:
    """Retrieve the best model for a given role, using cache if valid."""
    segments = _load_model_segments()
    if not segments:
        return DEFAULT_MODEL

    role_models = segments.get(role.upper(), [])
    if not role_models:
        return DEFAULT_MODEL

    return _format_model_name(role_models[0])