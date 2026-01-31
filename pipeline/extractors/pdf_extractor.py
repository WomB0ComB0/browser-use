from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from pipeline.extractors.base import BaseExtractor, ExtractedContent
from pipeline.extractors.ocr_extractor import OCRExtractor

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


class PdfStructure(TypedDict, total=False):
    """Structure information for PDF files."""
    page_count: int
    title: str | None
    author: str | None
    subject: str | None
    creator: str | None
    producer: str | None
    creation_date: str | None
    has_toc: bool
    is_encrypted: bool
    page_sizes: list[tuple[float, float]]


class PdfExtractor(BaseExtractor):
    """Extractor for PDF files."""

    SUPPORTED_EXTENSIONS = [".pdf"]

    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the given file."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract content from a PDF file."""
        if not PYPDF_AVAILABLE:
            return self._create_unavailable_result(file_path)

        file_path = Path(file_path)
        metadata = self._get_file_metadata(file_path)

        try:
            reader = PdfReader(file_path)

            # Extract text from all pages
            text_content = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)

            content = "\n\n".join(text_content)

            # Check if likely scanned (very little text)
            if len(content.strip()) < 50:  # Heuristic: < 50 chars total
                # Try OCR
                try:
                    ocr_extractor = OCRExtractor()
                    ocr_result = ocr_extractor.extract(file_path)
                    if ocr_result and len(ocr_result.content) > len(content):
                        return ocr_result
                except Exception:
                    # Log but fall back to original empty content
                    pass

            # Extract metadata
            pdf_metadata = reader.metadata
            structure = self._build_structure(reader, pdf_metadata)

            # Generate summary
            summary = self._generate_summary(content, structure)

            return ExtractedContent(
                content=content,
                summary=summary,
                file_path=file_path,
                file_type="PDF",
                file_size_bytes=metadata["file_size_bytes"],
                modified_time=metadata["modified_time"],
                structure=structure,
                metadata={
                    "page_count": structure.get("page_count", 0),
                    "is_encrypted": structure.get("is_encrypted", False),
                },
            )

        except Exception as e:
            return ExtractedContent(
                content=f"Error extracting PDF: {e}",
                summary=f"Failed to extract PDF content: {e}",
                file_path=file_path,
                file_type="PDF",
                file_size_bytes=metadata["file_size_bytes"],
                modified_time=metadata["modified_time"],
                structure=None,
                metadata={"error": str(e)},
            )

    def _build_structure(self, reader: PdfReader, pdf_metadata: dict[str, Any] | None) -> PdfStructure:
        """Build structure information from PDF metadata."""
        structure: PdfStructure = {
            "page_count": len(reader.pages),
            "is_encrypted": reader.is_encrypted,
            "has_toc": bool(reader.outline),
        }

        if pdf_metadata:
            structure["title"] = pdf_metadata.get("/Title")
            structure["author"] = pdf_metadata.get("/Author")
            structure["subject"] = pdf_metadata.get("/Subject")
            structure["creator"] = pdf_metadata.get("/Creator")
            structure["producer"] = pdf_metadata.get("/Producer")

            creation_date = pdf_metadata.get("/CreationDate")
            if creation_date:
                structure["creation_date"] = str(creation_date)

        # Get page sizes (first 5 pages)
        page_sizes = []
        for page in reader.pages[:5]:
            box = page.mediabox
            page_sizes.append((float(box.width), float(box.height)))
        structure["page_sizes"] = page_sizes

        return structure

    def _generate_summary(self, content: str, structure: PdfStructure) -> str:
        """Generate a summary of the PDF content."""
        page_count = structure.get("page_count", 0)
        title = structure.get("title", "Untitled")
        author = structure.get("author", "Unknown")

        word_count = len(content.split())
        char_count = len(content)

        summary_parts = [
            f"PDF document with {page_count} page(s)",
        ]

        if title and title != "Untitled":
            summary_parts.append(f"Title: {title}")
        if author and author != "Unknown":
            summary_parts.append(f"Author: {author}")

        summary_parts.append(f"{word_count:,} words, {char_count:,} characters extracted")

        if structure.get("has_toc"):
            summary_parts.append("Contains table of contents")

        if structure.get("is_encrypted"):
            summary_parts.append("Document is encrypted")

        return ". ".join(summary_parts) + "."

    def _create_unavailable_result(self, file_path: Path) -> ExtractedContent:
        """Create result when pypdf is not available."""
        metadata = self._get_file_metadata(file_path)
        return ExtractedContent(
            content="PDF extraction requires pypdf. Install with: pip install pypdf",
            summary="PDF extraction unavailable - pypdf not installed",
            file_path=file_path,
            file_type="PDF",
            file_size_bytes=metadata["file_size_bytes"],
            modified_time=metadata["modified_time"],
            structure=None,
            metadata={"error": "pypdf not installed"},
        )
