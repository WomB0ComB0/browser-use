from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import yaml

from pipeline.config import PipelineConfig
from pipeline.extractors import get_extractor_for_file
from pipeline.generators import BaseGenerator, get_generator
from pipeline.memory.pinecone_service import PineconeMemory
from pipeline.utils.logging import get_logger, setup_logging
from pipeline.utils.metrics import PipelineMetrics
from pipeline.watcher import FileWatcher

if TYPE_CHECKING:
    from logging import Logger

    from pipeline.orchestrator import AgentOrchestrator


class PipelineProcessor:
    """Main pipeline processor that coordinates extraction and generation."""
    
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger: Logger = get_logger(__name__)
        self.metrics = PipelineMetrics()
        self.generator: BaseGenerator | None = None
        self.memory: PineconeMemory | None = None
        self.watcher: FileWatcher | None = None
        self.orchestrator: AgentOrchestrator | None = None
        self._processing_queue: asyncio.Queue[Path] = asyncio.Queue()
        self._shutdown: bool = False
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
        
        # Initialize generator using factory
        self.generator = get_generator(self.config)
        
        # Initialize orchestrator
        from pipeline.orchestrator import AgentOrchestrator
        self.orchestrator = AgentOrchestrator(self.config)
        
        # Initialize memory
        self.memory = PineconeMemory(self.config)
        if self.memory.enabled:
            self.memory._initialize()  # Pre-initialize to check connection

        
        self.logger.info("Pipeline initialized")
    
    async def process_file(self, file_path: Path, workflow_name: str | None = None) -> bool:
        """Process a single file through the pipeline.
        
        Args:
            file_path: Path to the file to process.
            workflow_name: Optional name of special workflow to run.
            
        Returns:
            True if processing succeeded, False otherwise.
        """
        self.logger.info(f"Processing: {file_path}")
        if workflow_name:
            self.logger.info(f"Using workflow: {workflow_name}")
        
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
            
            # Run workflow or generation
            output_content, model_used = await self._run_workflow(content, workflow_name)
            
            # Save output
            output_path = await self._save_execution_output(file_path, workflow_name, output_content)
            self.logger.info(f"Generated: {output_path}")

            # Store in memory
            self._store_execution_memory(
                file_path=file_path, 
                workflow_name=workflow_name,
                content_type=content.file_type,
                output_content=output_content,
                output_path=output_path,
                model_used=model_used
            )

            self.metrics.end_processing(success=True)
            
            # Save metrics to disk so separate dashboard process can read them
            metrics_path = self.config.get_logs_dir() / "metrics.json"
            self.metrics.save(metrics_path)
            
            return True
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error processing {file_path}: {e}\n{traceback.format_exc()}")
            self.metrics.end_processing(success=False, error=str(e))
            return False

    async def _run_workflow(self, content, workflow_name: str | None) -> tuple[str, str]:
        """Run a workflow or standard generation."""
        if workflow_name and self.orchestrator:
            if workflow_name == "startup_application":
                wf = self.orchestrator.create_startup_application_workflow()
            elif workflow_name == "code_review":
                wf = self.orchestrator.create_code_review_workflow()
            else:
                # Try to load from YAML
                workflow_path = self.config._base_path / "pipeline" / "workflows" / f"{workflow_name}.yaml"
                if workflow_path.exists():
                    self.logger.info(f"Loading workflow from {workflow_path}")
                    async with aiofiles.open(workflow_path, mode='r') as f:
                        workflow_content = await f.read()
                        wf = yaml.safe_load(workflow_content)
                else:
                    raise ValueError(f"Unknown workflow: {workflow_name}")
            
            result = await self.orchestrator.execute_workflow(wf, content)
            return result.final_output, "multi-agent-workflow"
        
        # Standard generation
        if self.generator is None:
            raise RuntimeError("Generator not initialized")
        
        instr_result = await self.generator.generate(content)
        return instr_result.instructions, instr_result.model_used

    async def _save_execution_output(self, file_path: Path, workflow_name: str | None, content: str) -> Path:
        """Save execution output to a markdown file."""
        output_name = f"{file_path.stem}_{workflow_name if workflow_name else 'instructions'}.md"
        if not output_name.endswith(".md"):
            output_name += ".md"
        output_path = self.config.get_output_dir() / output_name
        
        # Manual write since GeneratedInstructions isn't returned by orchestrator yet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(output_path, mode="w", encoding="utf-8") as f:
            await f.write(f"# Processed: {file_path.name}\n\n{content}")
            
        return output_path

    def _store_execution_memory(
        self, 
        file_path: Path, 
        workflow_name: str | None, 
        content_type: str, 
        output_content: str, 
        output_path: Path, 
        model_used: str
    ) -> None:
        """Store execution results in Pinecone memory if enabled."""
        if getattr(self, "memory", None) and self.memory.enabled:
            self.logger.info(f"Storing in memory: {file_path.name}")
            self.memory.upsert(
                content=output_content,
                source_file=file_path.name,
                metadata={
                    "type": "workflow" if workflow_name else "instruction_generation",
                    "workflow_name": workflow_name,
                    "file_type": content_type,
                    "output_path": str(output_path),
                    "model": model_used
                }
            )
    
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
