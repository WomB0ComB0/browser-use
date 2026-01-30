"""Extractor package for different file types."""
from __future__ import annotations

from pathlib import Path

from pipeline.extractors.base import BaseExtractor, ExtractedContent
from pipeline.extractors.csv_extractor import CsvExtractor
from pipeline.extractors.excel_extractor import ExcelExtractor
from pipeline.extractors.json_extractor import JsonExtractor
from pipeline.extractors.pdf_extractor import PdfExtractor
from pipeline.extractors.text import TextExtractor

__all__ = [
    "BaseExtractor",
    "ExtractedContent",
    "TextExtractor",
    "JsonExtractor",
    "CsvExtractor",
    "PdfExtractor",
    "ExcelExtractor",
    "get_extractor_for_file",
]


def get_extractor_for_file(file_path: str | Path) -> BaseExtractor:
    """Get the appropriate extractor for a file based on its extension."""
    suffix = Path(file_path).suffix.lower()

    extractors: dict[str, type[BaseExtractor]] = {
        # Text formats
        ".txt": TextExtractor,
        ".md": TextExtractor,
        ".rst": TextExtractor,
        # Data formats
        ".json": JsonExtractor,
        ".csv": CsvExtractor,
        # Document formats
        ".pdf": PdfExtractor,
        ".xlsx": ExcelExtractor,
        ".xlsm": ExcelExtractor,
    }

    extractor_class = extractors.get(suffix, TextExtractor)
    return extractor_class()
