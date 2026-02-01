"""File system monitoring and event handling for the processing pipeline.

This module uses the Watchdog library to observe directories for new or
modified files, applying a debouncing mechanism to prevent redundant
processing of rapidly changing files.
"""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from pipeline.config import PipelineConfig
from pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from logging import Logger


class DebouncedHandler(FileSystemEventHandler):
    """Event handler that delays and debounces file system events.

    This prevents the pipeline from processing a file multiple times while it's
    still being written by the OS or another application.
    """
    
    def __init__(
        self,
        callback: Callable[[Path], None],
        config: PipelineConfig,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.callback = callback
        self.config = config
        self.loop = loop
        self.pending: dict[str, asyncio.TimerHandle] = {}
        self.logger: Logger = get_logger(__name__)
    
    def _should_ignore(self, path: str) -> bool:
        """Check if path matches any ignore patterns."""
        name = Path(path).name
        for pattern in self.config.watcher.ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
    
    def _is_supported(self, path: str) -> bool:
        """Check if file extension is supported."""
        suffix = Path(path).suffix.lower()
        return suffix in self.config.processing.supported_extensions
    
    def _schedule_callback(self, path: str) -> None:
        """Schedule a debounced callback for the path."""
        if path in self.pending:
            self.pending[path].cancel()
        
        def run_callback() -> None:
            del self.pending[path]
            # Fix: Race condition check - ensure file still exists
            if Path(path).exists():
                self.callback(Path(path))
        
        handle = self.loop.call_later(
            self.config.watcher.debounce_seconds,
            run_callback
        )
        self.pending[path] = handle
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return
        
        path = str(event.src_path)
        if self._should_ignore(path) or not self._is_supported(path):
            return
        
        self.logger.info(f"File created: {path}")
        self.loop.call_soon_threadsafe(self._schedule_callback, path)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        
        path = str(event.src_path)
        if self._should_ignore(path) or not self._is_supported(path):
            return
        
        self.logger.debug(f"File modified: {path}")
        self.loop.call_soon_threadsafe(self._schedule_callback, path)


class FileWatcher:
    """Wrapper around Watchdog Observer for pipeline-specific monitoring.

    Manages the lifecycle of directory watching, handles event dispatching via
    DebouncedHandler, and provides utility to scan for existing files.
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        callback: Callable[[Path], None],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.config = config
        self.callback = callback
        self.loop = loop or asyncio.get_event_loop()
        self.observer: Observer | None = None
        self.logger: Logger = get_logger(__name__)
    
    def start(self) -> None:
        """Start watching the data directory."""
        watch_path = self.config.get_data_dir()
        watch_path.mkdir(parents=True, exist_ok=True)
        
        handler = DebouncedHandler(self.callback, self.config, self.loop)
        
        self.observer = Observer()
        self.observer.schedule(
            handler,
            str(watch_path),
            recursive=self.config.watcher.recursive
        )
        self.observer.start()
        
        self.logger.info(f"Started watching: {watch_path}")
    
    def stop(self) -> None:
        """Stop watching the directory."""
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None
            self.logger.info("Stopped watching")
    
    def process_existing(self) -> list[Path]:
        """Find and return existing files in the data directory."""
        files = []
        data_dir = self.config.get_data_dir()
        
        for ext in self.config.processing.supported_extensions:
            pattern = f"**/*{ext}" if self.config.watcher.recursive else f"*{ext}"
            files.extend(data_dir.glob(pattern))
        
        return [f for f in files if f.is_file()]
