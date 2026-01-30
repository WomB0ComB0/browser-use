from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.config import PipelineConfig
from pipeline.extractors import get_extractor_for_file
from pipeline.generators import GeminiGenerator
from pipeline.utils.logging import get_logger, setup_logging
from pipeline.utils.metrics import PipelineMetrics
from pipeline.watcher import FileWatcher

if TYPE_CHECKING:
    from logging import Logger


class PipelineProcessor:
    """Main pipeline processor that coordinates extraction and generation."""
    
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger: Logger = get_logger(__name__)
        self.metrics = PipelineMetrics()
        self.generator: GeminiGenerator | None = None
        self.watcher: FileWatcher | None = None
        self._processing_queue: asyncio.Queue[Path] = asyncio.Queue()
        self._shutdown = False
        self._workers: list[asyncio.Task[None]] = []
    
    def initialize(self) -> None:
        """Initialize the pipeline components."""
        # Ensure directories exist
        self.config.ensure_directories()
        
        # Setup logging
        setup_logging(
            level=self.config.logging.level,
            format_type=self.config.logging.format,
            log_dir=self.config.get_logs_dir(),
        )
        
        self.logger.info("Initializing pipeline components")
        
        # Initialize generator
        self.generator = GeminiGenerator(self.config)
        
        self.logger.info("Pipeline initialized")
    
    async def process_file(self, file_path: Path) -> bool:
        """Process a single file through the pipeline.
        
        Args:
            file_path: Path to the file to process.
            
        Returns:
            True if processing succeeded, False otherwise.
        """
        self.logger.info(f"Processing: {file_path}")
        
        try:
            # Get appropriate extractor
            extractor = get_extractor_for_file(file_path)
            
            # Start metrics
            stat = file_path.stat()
            self.metrics.start_processing(
                file_path=file_path,
                file_type=extractor.__class__.__name__,
                file_size=stat.st_size,
            )
            
            # Extract content
            content = extractor.extract(file_path)
            self.logger.debug(f"Extracted: {content.file_type}, {len(content.content)} chars")
            
            # Generate instructions
            if self.generator is None:
                raise RuntimeError("Generator not initialized")
            
            instructions = await self.generator.generate(content)
            
            # Save output
            output_name = f"{file_path.stem}_instructions.md"
            output_path = self.config.get_output_dir() / output_name
            instructions.save(output_path)
            
            self.logger.info(f"Generated: {output_path}")
            self.metrics.end_processing(success=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            self.metrics.end_processing(success=False, error=str(e))
            return False
    
    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes files from the queue."""
        self.logger.debug(f"Worker {worker_id} started")
        
        while not self._shutdown:
            try:
                # Get file from queue with timeout
                try:
                    file_path = await asyncio.wait_for(
                        self._processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the file
                await self.process_file(file_path)
                self._processing_queue.task_done()
                
            except asyncio.CancelledError:
                self.logger.info(f"Worker {worker_id} cancelled")
                raise
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    def _on_file_change(self, file_path: Path) -> None:
        """Callback for file watcher events."""
        self.logger.debug(f"File change detected: {file_path}")
        
        # Add to processing queue
        try:
            self._processing_queue.put_nowait(file_path)
        except asyncio.QueueFull:
            self.logger.warning(f"Queue full, skipping: {file_path}")
    
    async def start(self, process_existing: bool = True) -> None:
        """Start the pipeline.
        
        Args:
            process_existing: Whether to process existing files in data folder.
        """
        self.initialize()
        
        self.logger.info("Starting pipeline...")
        
        # Start workers
        for i in range(self.config.processing.concurrent_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        # Setup file watcher
        loop = asyncio.get_event_loop()
        self.watcher = FileWatcher(self.config, self._on_file_change, loop)
        
        # Process existing files if requested
        if process_existing:
            existing = self.watcher.process_existing()
            self.logger.info(f"Found {len(existing)} existing files")
            for file_path in existing:
                await self._processing_queue.put(file_path)
        
        # Start watching
        self.watcher.start()
        
        self.logger.info(
            f"Pipeline running. Watching: {self.config.get_data_dir()}"
        )
    
    async def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self.logger.info("Stopping pipeline...")
        
        self._shutdown = True
        
        # Stop watcher
        if self.watcher:
            self.watcher.stop()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()
        
        # Save metrics
        metrics_path = self.config.get_logs_dir() / "metrics.json"
        self.metrics.save(metrics_path)
        
        self.logger.info("Pipeline stopped")
        print(self.metrics.print_summary())
    
    async def run(self) -> None:
        """Run the pipeline until interrupted."""
        await self.start()
        
        try:
            # Keep running until interrupted
            while not self._shutdown:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.info("Pipeline run cancelled")
            raise
        finally:
            await self.stop()
