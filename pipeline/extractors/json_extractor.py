"""Extraction logic for JSON files.

Parses JSON content, handles various encodings, and performs structural 
analysis to provide accurate metadata about the keys and types contained 
within the document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from pipeline.extractors.base import BaseExtractor, ExtractedContent


class JsonStructure(TypedDict, total=False):
    """Metadata representation of a JSON document's schema.

    Attributes:
        type: The fundamental type (object, array, string, etc.).
        truncated: True if the analysis reached max depth.
        keys: List of top-level keys (for objects).
        key_count: Number of keys in the object.
        sample_values: Nested JsonStructure for a subset of object values.
        length: Number of items (for arrays).
        item_types: Set of types found in the array.
        sample: JsonStructure for the first item in an array.
        value_preview: Truncated string representation of a primitive value.
    """
    type: str
    truncated: bool
    keys: list[str]
    key_count: int
    sample_values: dict[str, JsonStructure]
    length: int
    item_types: list[str]
    sample: JsonStructure | None
    value_preview: str | None


class JsonExtractor(BaseExtractor):
    """Handler for extracting content and structural insights from JSON files.

    Maintains the raw JSON structure but provides a summary and detailed 
    meta-analysis of the data schema to inform AI reasoning.
    """
    
    SUPPORTED_EXTENSIONS = {".json"}
    
    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the given file."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract content from a JSON file."""
        file_path = Path(file_path)
        metadata = self._get_file_metadata(file_path)
        
        # Attempt to read with common encodings
        content = None
        for encoding in ["utf-8", "latin-1"]:  # Add more encodings if needed
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break  # Successfully read, exit loop
            except UnicodeDecodeError:
                continue  # Try next encoding
        
        if content is None:
            # If all attempts fail, read as binary and decode with replacement
            with open(file_path, "rb") as f:
                content = f.read().decode("utf-8", errors="replace")

        # Parse JSON
        try:
            data = json.loads(content)
            parsed = True
        except json.JSONDecodeError:
            data = None
            parsed = False
        
        # Create formatted content for readability
        if parsed:
            formatted_content = json.dumps(data, indent=2)
        else:
            formatted_content = content
        
        # Analyze structure
        structure = self._analyze_structure(data) if parsed else None
        
        # Create summary
        summary = self._create_json_summary(data) if parsed else self._create_summary(content)
        
        return ExtractedContent(
            content=formatted_content,
            summary=summary,
            file_path=file_path,
            file_type="JSON",
            file_size_bytes=metadata["file_size_bytes"],
            modified_time=metadata["modified_time"],
            structure=structure,
            metadata={
                "parsed": parsed,
                "data_type": type(data).__name__ if parsed else "unknown",
            },
        )
    
    def _analyze_structure(self, data: Any, depth: int = 0, max_depth: int = 3) -> JsonStructure:
        """Analyze the structure of JSON data."""
        if depth >= max_depth:
            return {"type": type(data).__name__, "truncated": True}
        
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys())[:20],  # Limit keys shown
                "key_count": len(data),
                "sample_values": {
                    k: self._analyze_structure(v, depth + 1, max_depth)
                    for k, v in list(data.items())[:5]
                },
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "item_types": list({type(item).__name__ for item in data[:10]}),
                "sample": self._analyze_structure(data[0], depth + 1, max_depth) if data else None,
            }
        else:
            return {
                "type": type(data).__name__,
                "value_preview": str(data)[:100] if data else None,
            }
    
    def _create_json_summary(self, data: Any) -> str:
        """Create a summary for JSON data."""
        if isinstance(data, dict):
            keys = list(data.keys())[:5]
            key_str = ", ".join(keys)
            if len(data) > 5:
                key_str += f", ... ({len(data)} total keys)"
            return f"JSON object with keys: {key_str}"
        elif isinstance(data, list):
            if not data:
                return "Empty JSON array"
            item_type = type(data[0]).__name__
            return f"JSON array with {len(data)} {item_type} items"
        else:
            return f"JSON {type(data).__name__}: {str(data)[:200]}"

