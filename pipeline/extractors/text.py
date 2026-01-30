from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from pipeline.extractors.base import BaseExtractor, ExtractedContent


class MarkdownStructure(TypedDict):
    headers: list[dict[str, int | str]]
    code_blocks: int
    header_count: int


class TextExtractor(BaseExtractor):
    """Extractor for plain text and markdown files."""
    
    _MD_EXTS = {".md", ".markdown"}
    _TXT_EXTS = {".txt", ".text"}
    SUPPORTED_EXTENSIONS = _MD_EXTS | _TXT_EXTS | {".rst"}
    _MARKDOWN_EXTENSIONS = _MD_EXTS
    
    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the given file."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract content from a text file."""
        file_path = Path(file_path)
        metadata = self._get_file_metadata(file_path)
        
        # Try to detect encoding
        content = self._read_with_encoding_detection(file_path)
        
        # Determine file type description
        suffix = file_path.suffix.lower()
        if suffix in self._MD_EXTS:
            file_type = "Markdown"
        elif suffix in self._TXT_EXTS:
            file_type = "Plain Text"
        elif suffix == ".rst":
            file_type = "reStructuredText"
        else:
            file_type = "Text"
        
        # Create summary
        summary = self._create_summary(content)
        
        # Extract structure for markdown
        structure: MarkdownStructure | None = None
        if suffix in self._MARKDOWN_EXTENSIONS:
            structure = self._extract_markdown_structure(content)
        
        return ExtractedContent(
            content=content,
            summary=summary,
            file_path=file_path,
            file_type=file_type,
            file_size_bytes=metadata["file_size_bytes"],
            modified_time=metadata["modified_time"],
            structure=structure,
            metadata={
                "line_count": content.count("\n") + 1,
                "word_count": len(content.split()),
                "char_count": len(content),
            },
        )
    
    def _read_with_encoding_detection(self, file_path: Path) -> str:
        """Read file with encoding detection."""
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # Fallback: read as binary and decode with errors ignored
        with open(file_path, "rb") as bf:
            return bf.read().decode("utf-8", errors="ignore")
    
    def _extract_markdown_structure(self, content: str) -> MarkdownStructure:
        """Extract structure from markdown content."""
        headers: list[dict[str, int | str]] = []
        code_blocks = 0
        
        in_code_block = False
        for line in content.split("\n"):
            # Track code blocks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                if not in_code_block:
                    code_blocks += 1
                continue
            
            if in_code_block:
                continue
            
            # Extract headers
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("#").strip()
                if text:
                    headers.append({"level": level, "text": text})
        
        return {
            "headers": headers,
            "code_blocks": code_blocks,
            "header_count": len(headers),
        }
