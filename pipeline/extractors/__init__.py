"""Extractor package for different file types."""
from __future__ import annotations

from pathlib import Path

from pipeline.extractors.base import BaseExtractor, ExtractedContent
from pipeline.extractors.csv_extractor import CsvExtractor
from pipeline.extractors.json_extractor import JsonExtractor
from pipeline.extractors.text import TextExtractor

__all__ = [
    "BaseExtractor",
    "ExtractedContent",
    "TextExtractor",
    "JsonExtractor",
    "CsvExtractor",
]


def get_extractor_for_file(file_path: str | Path) -> BaseExtractor:
    """Get the appropriate extractor for a file based on its extension."""
    suffix = Path(file_path).suffix.lower()
    
    extractors: dict[str, type[BaseExtractor]] = {
        ".txt": TextExtractor,
        ".md": TextExtractor,
        ".rst": TextExtractor,
        ".json": JsonExtractor,
        ".csv": CsvExtractor,
    }
    
    extractor_class = extractors.get(suffix, TextExtractor)
    return extractor_class()
