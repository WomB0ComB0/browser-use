"""Base extractor interface and data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExtractedContent:
    """Container for extracted file content and metadata."""
    
    # Core content
    content: str
    summary: str
    
    # File metadata
    file_path: Path
    file_type: str
    file_size_bytes: int
    modified_time: datetime
    
    # Extracted structure (optional)
    structure: Optional[dict[str, Any]] = None
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def file_name(self) -> str:
        """Get the file name without path."""
        return self.file_path.name
    
    @property
    def extension(self) -> str:
        """Get the file extension."""
        return self.file_path.suffix.lower()


class BaseExtractor(ABC):
    """Abstract base class for file content extractors."""
    
    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract content from a file.
        
        Args:
            file_path: Path to the file to extract from.
            
        Returns:
            ExtractedContent with the file's content and metadata.
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the given file.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this extractor can handle the file.
        """
        pass
    
    def _get_file_metadata(self, file_path: Path) -> dict[str, Any]:
        """Get common file metadata."""
        stat = file_path.stat()
        return {
            "file_size_bytes": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
            "created_time": datetime.fromtimestamp(stat.st_ctime),
        }
    
    def _create_summary(self, content: str, max_length: int = 500) -> str:
        """Create a summary of the content."""
        lines = content.strip().split("\n")
        
        # Get first few non-empty lines
        summary_lines = []
        char_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if char_count + len(line) > max_length:
                break
            summary_lines.append(line)
            char_count += len(line)
        
        summary = " ".join(summary_lines)
        if len(content) > max_length:
            summary += "..."
        
        return summary
