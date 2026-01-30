from __future__ import annotations

import asyncio
import signal
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from pipeline.config import PipelineConfig
from pipeline.extractors import get_extractor_for_file
from pipeline.processor import PipelineProcessor

# Load environment variables
load_dotenv()

app = typer.Typer(
    name="pipeline",
    help="Enterprise Data Processing Pipeline",
    add_completion=False,
)
console = Console()
_CONFIG_HELP = "Path to config file"


def get_config(config_path: str | None = None) -> PipelineConfig:
    """Load configuration."""
    base_path = Path(__file__).parent.parent
    return PipelineConfig.load(config_path, base_path)


@app.command()
def start(
    config: str | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    no_existing: bool = typer.Option(False, "--no-existing", help="Don't process existing files"),
    watch: bool = typer.Option(True, "--watch", help="Watch for changes (always enabled for start)"),
):
    """Start the pipeline and watch for new files."""
    cfg = get_config(config)
    
    console.print(Panel.fit(
        "[bold green]Starting Enterprise Data Pipeline[/bold green]\n"
        f"Watching: [cyan]{cfg.get_data_dir()}[/cyan]\n"
        f"Output: [cyan]{cfg.get_output_dir()}[/cyan]",
        title="Pipeline",
    ))
    
    processor = PipelineProcessor(cfg)
    
    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(processor.stop()))
    
    try:
        loop.run_until_complete(processor.start(process_existing=not no_existing))
        loop.run_until_complete(processor.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        loop.run_until_complete(processor.stop())
        loop.close()


@app.command()
def process(
    file: Path = typer.Argument(..., help="File to process"),
    config: str | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    workflow: str | None = typer.Option(None, "--workflow", "-w", help="Workflow to run"),
):
    """Process a single file."""
    if not file.exists():
        console.print(f"[red]Error: File not found: {file}[/red]")
        raise typer.Exit(1)
    
    cfg = get_config(config)
    processor = PipelineProcessor(cfg)
    
    async def run() -> bool:
        processor.initialize()
        success = await processor.process_file(file, workflow_name=workflow)
        return success
    
    success = asyncio.run(run())
    
    if success:
        console.print(f"[green]✓ Successfully processed: {file}[/green]")
        output_name = f"{file.stem}_instructions.md"
        console.print(f"  Output: [cyan]{cfg.get_output_dir() / output_name}[/cyan]")
    else:
        console.print(f"[red]✗ Failed to process: {file}[/red]")
        raise typer.Exit(1)


@app.command("config")
def show_config(
    config: str | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
):
    """Display current configuration."""
    cfg = get_config(config)
    
    console.print(Panel.fit(
        f"""[bold]Directories[/bold]
  Data:   [cyan]{cfg.directories.data}[/cyan]
  Output: [cyan]{cfg.directories.output}[/cyan]
  Logs:   [cyan]{cfg.directories.logs}[/cyan]

[bold]Processing[/bold]
  Extensions: {', '.join(cfg.processing.supported_extensions)}
  Workers:    {cfg.processing.concurrent_workers}
  Max Size:   {cfg.processing.max_file_size_mb} MB

[bold]Generator[/bold]
  Model:       [cyan]{cfg.generator.model}[/cyan]
  Temperature: {cfg.generator.temperature}
  Max Tokens:  {cfg.generator.max_tokens}

[bold]Logging[/bold]
  Level:  {cfg.logging.level}
  Format: {cfg.logging.format}""",
        title="Pipeline Configuration",
    ))


@app.command()
def status(
    config: str | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
):
    """Show pipeline status and metrics."""
    cfg = get_config(config)
    
    metrics_file = cfg.get_logs_dir() / "metrics.json"
    
    if not metrics_file.exists():
        console.print("[yellow]No metrics found. Pipeline may not have run yet.[/yellow]")
        return
    
    import json
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    console.print(Panel.fit(
        f"""[bold]Files Processed[/bold]
  Total:     {metrics['files_processed']}
  Succeeded: [green]{metrics['files_succeeded']}[/green]
  Failed:    [red]{metrics['files_failed']}[/red]

[bold]Performance[/bold]
  Success Rate: {metrics['success_rate_percent']}%
  Avg Time:     {metrics['average_processing_time_seconds']}s
  Data:         {metrics['total_bytes_processed']:,} bytes

[bold]Runtime[/bold]
  Started: {metrics['started_at']}
  Uptime:  {metrics['uptime_seconds']}s""",
        title="Pipeline Status",
    ))


@app.command()
def dashboard(
    config: str | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind to"),
):
    """Start the web dashboard for real-time monitoring."""
    cfg = get_config(config)

    try:
        from pipeline.dashboard import DashboardApp
    except ImportError:
        console.print("[red]Dashboard requires FastAPI. Install with:[/red]")
        console.print("  pip install fastapi uvicorn websockets")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold green]Starting Pipeline Dashboard[/bold green]\n"
        f"URL: [cyan]http://{host}:{port}[/cyan]\n"
        f"Press Ctrl+C to stop",
        title="Dashboard",
    ))

    dashboard_app = DashboardApp(cfg)
    dashboard_app.run(host=host, port=port)
@app.command()
def browser_apply(
    file: Path = typer.Argument(..., help="File to derive info from (e.g. pitch deck)"),
    url: str = typer.Argument(..., help="URL of the form to fill"),
    config: str | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
):
    """Automatically derive info from a file and fill out a web form."""
    if not file.exists():
        console.print(f"[red]Error: File not found: {file}[/red]")
        raise typer.Exit(1)
    
    cfg = get_config(config)
    processor = PipelineProcessor(cfg)
    
    async def run():
        processor.initialize()
        console.print(f"[yellow]1. Extracting data from {file.name}...[/yellow]")
        
        # Run startup application workflow
        extractor = get_extractor_for_file(file)
        content = extractor.extract(file)
        
        if not processor.orchestrator:
            console.print("[red]Error: Orchestrator not initialized[/red]")
            return False
            
        wf = processor.orchestrator.create_startup_application_workflow()
        result = await processor.orchestrator.execute_workflow(wf, content)
        
        if not result.success:
            console.print("[red]Error: Workflow failed to extract data[/red]")
            return False
            
        # Extract JSON data
        data = processor.orchestrator.extract_json_from_output(result.final_output)
        if not data:
            console.print("[red]Error: Could not extract structured form data from AI output[/red]")
            console.print(f"Final output was: {result.final_output[:200]}...")
            return False
            
        console.print(f"[green]✓ Data extracted for startup: {data.get('startup_name', 'Unknown')}[/green]")
        console.print(f"[yellow]2. Starting browser agent to fill form at {url}...[/yellow]")
        
        # Initialize browser executor
        from pipeline.utils.browser_executor import BrowserExecutor
        executor = BrowserExecutor(cfg)
        
        try:
            exec_result = await executor.fill_form(url, data)
            console.print("[bold green]✓ Browser agent finished![/bold green]")
            console.print(f"Result: {exec_result}")
            return True
        except Exception as e:
            console.print(f"[red]Error during browser automation: {e}[/red]")
            return False

    success = asyncio.run(run())
    if not success:
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
