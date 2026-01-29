"""Text file extractor for .txt, .md, .rst files."""

from pathlib import Path

from pipeline.extractors.base import BaseExtractor, ExtractedContent


class TextExtractor(BaseExtractor):
    """Extractor for plain text and markdown files."""
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".text", ".markdown"}
    
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
        file_type_map = {
            ".txt": "Plain Text",
            ".md": "Markdown",
            ".markdown": "Markdown",
            ".rst": "reStructuredText",
            ".text": "Plain Text",
        }
        file_type = file_type_map.get(suffix, "Text")
        
        # Create summary
        summary = self._create_summary(content)
        
        # Extract structure for markdown
        structure = None
        if suffix in {".md", ".markdown"}:
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
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Fallback: read as binary and decode with errors ignored
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    
    def _extract_markdown_structure(self, content: str) -> dict:
        """Extract structure from markdown content."""
        headers = []
        code_blocks = 0
        links = []
        
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
