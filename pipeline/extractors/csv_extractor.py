from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import TypedDict

from pipeline.extractors.base import BaseExtractor, ExtractedContent, FileMetadata


class ColumnInfo(TypedDict):
    name: str
    type: str
    non_null_count: int
    null_count: int
    unique_count: int
    sample_values: list[str]


class CsvStructure(TypedDict):
    headers: list[str]
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    preview: str


class CsvExtractor(BaseExtractor):
    """Extractor for CSV files."""
    
    SUPPORTED_EXTENSIONS = {".csv", ".tsv"}
    
    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the given file."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def extract(self, file_path: Path) -> ExtractedContent:
        """Extract content from a CSV file."""
        file_path = Path(file_path)
        metadata = self._get_file_metadata(file_path)
        
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            content = f.read()
        
        # Parse CSV
        delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
        
        try:
            # Detect dialect
            sample = content[:4096]
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            pass
        
        # Read CSV data
        reader = csv.reader(StringIO(content), delimiter=delimiter)
        rows = list(reader)
        
        if not rows:
            return self._create_empty_result(file_path, metadata, content)
        
        # Extract structure
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []
        
        # Analyze columns
        column_info = self._analyze_columns(headers, data_rows)
        
        # Create summary
        summary = self._create_csv_summary(headers, data_rows)
        
        # Create a formatted preview
        preview = self._create_preview(headers, data_rows[:10])
        
        return ExtractedContent(
            content=content,
            summary=summary,
            file_path=file_path,
            file_type="CSV" if delimiter == "," else "TSV",
            file_size_bytes=metadata["file_size_bytes"],
            modified_time=metadata["modified_time"],
            structure=CsvStructure(
                headers=headers,
                row_count=len(data_rows),
                column_count=len(headers),
                columns=column_info,
                preview=preview,
            ),
            metadata={
                "delimiter": delimiter,
                "has_headers": True,
            },
        )
    
    def _create_empty_result(self, file_path: Path, metadata: FileMetadata, content: str) -> ExtractedContent:
        """Create result for empty CSV."""
        return ExtractedContent(
            content=content,
            summary="Empty CSV file",
            file_path=file_path,
            file_type="CSV",
            file_size_bytes=metadata["file_size_bytes"],
            modified_time=metadata["modified_time"],
            structure=CsvStructure(
                headers=[],
                row_count=0,
                column_count=0,
                columns=[],
                preview="",
            ),
            metadata={},
        )
    
    def _analyze_columns(self, headers: list[str], data_rows: list[list[str]]) -> list[ColumnInfo]:
        """Analyze column types and statistics."""
        columns: list[ColumnInfo] = []
        
        for i, header in enumerate(headers):
            col_values = [row[i] for row in data_rows if i < len(row)]
            
            # Detect type
            col_type = self._detect_column_type(col_values)
            
            # Calculate statistics
            non_empty = [v for v in col_values if v.strip()]
            
            columns.append({
                "name": header,
                "type": col_type,
                "non_null_count": len(non_empty),
                "null_count": len(col_values) - len(non_empty),
                "unique_count": len(set(non_empty)),
                "sample_values": list(set(non_empty))[:5],
            })
        
        return columns
    
    def _detect_column_type(self, values: list[str]) -> str:
        """Detect the type of values in a column."""
        non_empty = [v for v in values if v.strip()]
        
        if not non_empty:
            return "empty"
        
        # Check if numeric
        numeric_count = 0
        int_count = 0
        
        for v in non_empty[:100]:  # Sample first 100
            try:
                float(v.replace(",", ""))
                numeric_count += 1
                if "." not in v:
                    int_count += 1
            except ValueError:
                pass
        
        if numeric_count == len(non_empty[:100]):
            return "integer" if int_count == numeric_count else "number"
        
        # Check if boolean
        bool_values = {"true", "false", "yes", "no", "1", "0", "y", "n"}
        if all(v.lower() in bool_values for v in non_empty[:100]):
            return "boolean"
        
        return "string"
    
    def _create_csv_summary(self, headers: list[str], data_rows: list[list[str]]) -> str:
        """Create a summary for CSV data."""
        col_str = ", ".join(headers[:5])
        if len(headers) > 5:
            col_str += f", ... ({len(headers)} total columns)"
        return f"CSV with {len(data_rows)} rows. Columns: {col_str}"
    
    def _create_preview(self, headers: list[str], sample_rows: list[list[str]]) -> str:
        """Create a text preview of the CSV."""
        lines = [",".join(headers)]
        for row in sample_rows[:5]:
            lines.append(",".join(row))
        return "\n".join(lines)
