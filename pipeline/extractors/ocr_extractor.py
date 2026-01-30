from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from pipeline.extractors.base import BaseExtractor, ExtractedContent
from pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from logging import Logger

    from easyocr import Reader


class OCRExtractor(BaseExtractor):
    """Extractor using EasyOCR for scanned documents."""

    def __init__(self, languages: list[str] | None = None) -> None:
        self.logger: Logger = get_logger(__name__)
        self._reader: Reader | None = None
        self._languages: list[str] = languages or ['en']

    def _load_model(self) -> None:
        """Lazy load the EasyOCR reader."""
        if self._reader is None and EASYOCR_AVAILABLE:
            self.logger.info(f"Loading EasyOCR model for languages: {self._languages}")
            try:
                # gpu=False by default to be "forgiving" unless CUDA is clearly available, 
                # but EasyOCR auto-detects. Let's let it auto-detect but we could force gpu=False if requested.
                # Given "more forgiving", usage of GPU is fine if available, but it handles CPU well.
                self._reader = easyocr.Reader(self._languages)
            except Exception as e:
                self.logger.error(f"Failed to load EasyOCR: {e}")
                raise

    def supports(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file."""
        if not EASYOCR_AVAILABLE:
            return False
        return file_path.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract content using OCR."""
        metadata = self._get_file_metadata(file_path)
        
        if not EASYOCR_AVAILABLE:
             return self._create_error_result(file_path, metadata, "EasyOCR not installed. Run `pip install easyocr`.")

        try:
            self._load_model()
            
            images = []
            if file_path.suffix.lower() == ".pdf":
                if not PDF2IMAGE_AVAILABLE:
                    return self._create_error_result(file_path, metadata, "pdf2image not installed.")
                images = convert_from_path(str(file_path))
            else:
                images = [Image.open(file_path).convert("RGB")]

            extracted_text_parts = []
            
            for i, image in enumerate(images):
                # EasyOCR expects numpy array or file path
                img_np = np.array(image)
                
                # detail=0 returns just the text list
                text_list = self._reader.readtext(img_np, detail=0)
                page_text = "\n".join(text_list)
                
                extracted_text_parts.append(f"--- Page {i+1} ---\n{page_text}")

            content = "\n\n".join(extracted_text_parts)
            summary = self._create_summary(content)

            return ExtractedContent(
                content=content,
                summary=summary,
                file_path=file_path,
                file_type="PDF (OCR)" if file_path.suffix.lower() == ".pdf" else "Image (OCR)",
                file_size_bytes=metadata["file_size_bytes"],
                modified_time=metadata["modified_time"],
                metadata={"ocr_engine": "EasyOCR"}
            )

        except Exception as e:
            self.logger.error(f"OCR extraction failed for {file_path}: {e}")
            return self._create_error_result(file_path, metadata, str(e))

    def _create_error_result(self, file_path: Path, metadata: dict[str, Any], error_msg: str) -> ExtractedContent:
        return ExtractedContent(
            content=f"Error extracting content via OCR: {error_msg}",
            summary=f"Failed to extract content: {error_msg}",
            file_path=file_path,
            file_type="Unknown",
            file_size_bytes=metadata["file_size_bytes"],
            modified_time=metadata["modified_time"],
            metadata={"error": error_msg},
        )
