"""Pipeline metrics collection."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ProcessingRecord:
    """Record of a single file processing."""
    file_path: str
    file_type: str
    file_size: int
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class PipelineMetrics:
    """Collect and report pipeline metrics."""
    
    # Counters
    files_processed: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    total_bytes_processed: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    
    # Processing records
    records: list[ProcessingRecord] = field(default_factory=list)
    
    # Current processing
    _current: Optional[ProcessingRecord] = field(default=None, repr=False)
    
    def start_processing(self, file_path: Path, file_type: str, file_size: int) -> None:
        """Mark the start of file processing."""
        self._current = ProcessingRecord(
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_size,
            start_time=time.time(),
        )
    
    def end_processing(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark the end of file processing."""
        if self._current is None:
            return
        
        self._current.end_time = time.time()
        self._current.success = success
        self._current.error = error
        
        self.files_processed += 1
        self.total_bytes_processed += self._current.file_size
        
        if success:
            self.files_succeeded += 1
        else:
            self.files_failed += 1
        
        self.records.append(self._current)
        self._current = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.files_processed == 0:
            return 0.0
        return (self.files_succeeded / self.files_processed) * 100
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time in seconds."""
        durations = [r.duration_seconds for r in self.records if r.duration_seconds]
        if not durations:
            return 0.0
        return sum(durations) / len(durations)
    
    @property
    def uptime_seconds(self) -> float:
        """Get pipeline uptime in seconds."""
        return (datetime.now() - self.started_at).total_seconds()
    
    def get_summary(self) -> dict:
        """Get a summary of all metrics."""
        return {
            "files_processed": self.files_processed,
            "files_succeeded": self.files_succeeded,
            "files_failed": self.files_failed,
            "success_rate_percent": round(self.success_rate, 2),
            "total_bytes_processed": self.total_bytes_processed,
            "average_processing_time_seconds": round(self.average_processing_time, 3),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "started_at": self.started_at.isoformat(),
        }
    
    def save(self, output_path: Path) -> None:
        """Save metrics to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.get_summary()
        data["records"] = [
            {
                "file_path": r.file_path,
                "file_type": r.file_type,
                "file_size": r.file_size,
                "duration_seconds": r.duration_seconds,
                "success": r.success,
                "error": r.error,
            }
            for r in self.records[-100:]  # Keep last 100 records
        ]
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self) -> str:
        """Get a formatted summary for display."""
        summary = self.get_summary()
        lines = [
            "═" * 50,
            "Pipeline Metrics Summary",
            "═" * 50,
            f"Files Processed:  {summary['files_processed']}",
            f"  ✓ Succeeded:    {summary['files_succeeded']}",
            f"  ✗ Failed:       {summary['files_failed']}",
            f"Success Rate:     {summary['success_rate_percent']}%",
            f"Data Processed:   {self._format_bytes(summary['total_bytes_processed'])}",
            f"Avg Time:         {summary['average_processing_time_seconds']}s",
            f"Uptime:           {self._format_duration(summary['uptime_seconds'])}",
            "═" * 50,
        ]
        return "\n".join(lines)
    
    def _format_bytes(self, size: int) -> str:
        """Format bytes to human readable."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds to human readable."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"
