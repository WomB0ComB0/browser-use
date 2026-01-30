"""Enterprise Data Pipeline Package."""
from __future__ import annotations

from pipeline.config import PipelineConfig
from pipeline.processor import PipelineProcessor
from pipeline.watcher import FileWatcher

__version__ = "1.0.0"
__all__ = ["PipelineConfig", "PipelineProcessor", "FileWatcher"]
