"""File extractor factory and supported formats.

This package provides a central factory function and imports for various 
file format extractors including text, PDF, Excel, and JSON.
"""
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
    """Factory function to return the correct extractor for a file.

    Selects an extractor class based on the file extension. Defaults to
    TextExtractor if the extension is unknown.

    Args:
        file_path: Path to the target file.

    Returns:
        An instance of a class derived from BaseExtractor.
    """
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
