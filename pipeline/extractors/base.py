"""Base classes and types for all file extractors.

This module defines the interface that all specialized extractors (PDF, Excel, 
CSV, etc.) must implement, along with standardized containers for extracted
data and metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict


class FileMetadata(TypedDict):
    """Encapsulation of common file system attributes.

    Attributes:
        file_size_bytes: Size of the file in bytes.
        modified_time: Last modification timestamp.
        created_time: Creation timestamp.
    """
    file_size_bytes: int
    modified_time: datetime
    created_time: datetime


@dataclass
class ExtractedContent:
    """Standardized container for data extracted from a file.

    Attributes:
        content: The primary text content extracted from the file.
        summary: A brief snippet or summary of the content for quick preview.
        file_path: Absolute path to the source file.
        file_type: String identifier for the file format (e.g., 'PDF', 'CSV').
        file_size_bytes: Size of the source file.
        modified_time: Timestamp of last file modification.
        structure: Optional dictionary representing hierarchical or tabular data.
        metadata: Additional extractor-specific or system metadata.
    """
    
    # Core content
    content: str
    summary: str
    
    # File metadata
    file_path: Path
    file_type: str
    file_size_bytes: int
    modified_time: datetime
    
    # Extracted structure (optional)
    structure: dict[str, Any] | None = None
    
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
    """Abstract base class for all file extraction implementations.

    Provides common utilities for metadata gathering and summary creation
    shared across different file format handlers.
    """
    
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
    
    def _get_file_metadata(self, file_path: Path) -> FileMetadata:
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
