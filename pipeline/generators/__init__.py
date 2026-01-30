"""Instruction generator package."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.generators.base import BaseGenerator, GeneratedInstructions
from pipeline.generators.gemini import GeminiGenerator
from pipeline.generators.ollama import OllamaGenerator

if TYPE_CHECKING:
    from pipeline.config import PipelineConfig

__all__ = [
    "BaseGenerator",
    "GeneratedInstructions",
    "GeminiGenerator",
    "OllamaGenerator",
    "get_generator",
]


def get_generator(config: PipelineConfig) -> BaseGenerator:
    """Factory function to create the appropriate generator based on config.

    Args:
        config: Pipeline configuration.

    Returns:
        A generator instance (GeminiGenerator or OllamaGenerator).

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.generator.provider.lower()

    if provider == "gemini":
        return GeminiGenerator(config)
    elif provider == "ollama":
        return OllamaGenerator(config)
    else:
        raise ValueError(
            f"Unsupported generator provider: '{provider}'. "
            "Supported providers: 'gemini', 'ollama'"
        )
