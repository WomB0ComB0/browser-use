from __future__ import annotations

import asyncio
import signal
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from pipeline.config import PipelineConfig
from pipeline.processor import PipelineProcessor

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
):
    """Process a single file."""
    if not file.exists():
        console.print(f"[red]Error: File not found: {file}[/red]")
        raise typer.Exit(1)
    
    cfg = get_config(config)
    processor = PipelineProcessor(cfg)
    
    async def run() -> bool:
        processor.initialize()
        success = await processor.process_file(file)
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


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
