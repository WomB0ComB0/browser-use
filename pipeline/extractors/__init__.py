"""Extractor package for different file types."""

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


def get_extractor_for_file(file_path) -> BaseExtractor:
    """Get the appropriate extractor for a file based on its extension."""
    from pathlib import Path
    
    suffix = Path(file_path).suffix.lower()
    
    extractors = {
        ".txt": TextExtractor,
        ".md": TextExtractor,
        ".rst": TextExtractor,
        ".json": JsonExtractor,
        ".csv": CsvExtractor,
    }
    
    extractor_class = extractors.get(suffix, TextExtractor)
    return extractor_class()
