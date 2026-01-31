from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.config import PipelineConfig
from pipeline.utils.logging import get_logger
from pipeline.utils.metrics import PipelineMetrics

if TYPE_CHECKING:
    from logging import Logger

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class BroadcastLogHandler(logging.Handler):
    """Log handler that broadcasts logs to the dashboard via WebSocket."""
    
    def __init__(self, app: DashboardApp) -> None:
        super().__init__()
        self.app = app
        self.setLevel(logging.INFO)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            
            # Fire and forget mechanism to avoid blocking
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.app.broadcast_update("log", log_entry))
            except RuntimeError:
                # No event loop running
                pass
                
        except Exception:
            self.handleError(record)


class DashboardApp:
    """Real-time monitoring dashboard for the pipeline."""

    def __init__(self, config: PipelineConfig, metrics: PipelineMetrics | None = None) -> None:
        self.config = config
        self.metrics = metrics or PipelineMetrics()
        self.logger: Logger = get_logger(__name__)
        self._websocket_clients: list[WebSocket] = []

        if not FASTAPI_AVAILABLE:
            self.logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")
            self.app = None
            return

        self.app = FastAPI(
            title="Pipeline Dashboard",
            description="Real-time monitoring for the data processing pipeline",
            version="1.0.0",
        )

        self._setup_routes()
        self._setup_static_files()
        self._setup_log_streaming()

    def _setup_log_streaming(self) -> None:
        """Attach the broadcast log handler to the root logger."""
        handler = BroadcastLogHandler(self)
        logging.getLogger().addHandler(handler)
        self.logger.info("Dashboard log streaming enabled")

    def _setup_routes(self) -> None:
        """Set up API routes."""
        if not self.app:
            return

        self.app.get("/", response_class=HTMLResponse)(self._route_root)
        self.app.get("/api/health")(self._route_health)
        self.app.get("/api/metrics")(self._route_metrics)
        self.app.get("/api/history")(self._route_history)
        self.app.get("/api/config")(self._route_config)
        self.app.websocket("/ws")(self._route_websocket)

    async def _route_root(self) -> str:
        """Serve the dashboard HTML."""
        return self._get_dashboard_html()

    async def _route_health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }

    async def _route_metrics(self) -> dict:
        """Get current metrics."""
        # Reload from disk to catch updates from the processor
        metrics_path = self.config.get_logs_dir() / "metrics.json"
        if metrics_path.exists():
            self.metrics = PipelineMetrics.from_file(metrics_path)
        
        summary = self.metrics.get_summary()
        return {
            "files_processed": summary.get("files_processed", 0),
            "files_failed": summary.get("files_failed", 0),
            "success_rate": summary.get("success_rate_percent", 0.0) / 100.0,
            "total_processing_time": summary.get("total_processing_time", 0.0),
            "avg_processing_time": summary.get("average_processing_time_seconds", 0.0),
            "total_tokens": summary.get("total_tokens", 0),
        }

    async def _route_history(self) -> dict:
        """Get processing history."""
        # Ensure metrics are fresh
        metrics_path = self.config.get_logs_dir() / "metrics.json"
        if metrics_path.exists():
            self.metrics = PipelineMetrics.from_file(metrics_path)
            
        history = [
            {
                "file_name": Path(r.file_path).name,
                "success": r.success,
                "processing_time": r.duration_seconds or 0.0,
            }
            for r in reversed(self.metrics.records)
        ]
        return {
            "items": history,
            "total": len(history),
        }

    async def _route_config(self) -> dict:
        """Get current configuration (safe subset)."""
        return {
            "provider": self.config.generator.provider,
            "model": self.config.generator.model,
            "data_dir": str(self.config.directories.data),
            "output_dir": str(self.config.directories.output),
            "log_level": self.config.logging.level,
        }

    async def _route_websocket(self, websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time updates."""
        await websocket.accept()
        self._websocket_clients.append(websocket)
        # Send initial connection success message
        await websocket.send_text(json.dumps({
            "type": "log",
            "data": {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "level": "INFO",
                "logger": "Dashboard",
                "message": "Connected to real-time log stream"
            }
        }))
        self.logger.info("WebSocket client connected")

        try:
            while True:
                # Keep connection alive and receive any client messages
                data = await websocket.receive_text()
                # Handle client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            if websocket in self._websocket_clients:
                self._websocket_clients.remove(websocket)
            self.logger.info("WebSocket client disconnected")

    def _setup_static_files(self) -> None:
        """Mount static files directory."""
        if not self.app:
            return

        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    async def broadcast_update(self, event_type: str, data: dict) -> None:
        """Broadcast an update to all connected WebSocket clients."""
        message = json.dumps({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })

        # Filter out closed connections before sending
        active_clients = []
        for client in self._websocket_clients:
            try:
                await client.send_text(message)
                active_clients.append(client)
            except Exception:
                # Connection likely closed
                pass
        self._websocket_clients = active_clients

    async def notify_file_processed(
        self, file_name: str, success: bool, processing_time: float
    ) -> None:
        """Notify clients when a file has been processed."""
        await self.broadcast_update("file_processed", {
            "file_name": file_name,
            "success": success,
            "processing_time": processing_time,
        })

    async def notify_metrics_update(self) -> None:
        """Send updated metrics to all clients."""
        summary = self.metrics.get_summary()
        await self.broadcast_update("metrics_update", summary)

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run the dashboard server."""
        if not FASTAPI_AVAILABLE or not self.app:
            self.logger.error("Cannot run dashboard: FastAPI not available")
            return

        self.logger.info(f"Starting dashboard at http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

    def _get_dashboard_html(self) -> str:
        """Generate the dashboard HTML."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Dashboard</title>
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a25;
            --text-primary: #ffffff;
            --text-secondary: #9ca3af;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --success: #10b981;
            --error: #ef4444;
            --warning: #f59e0b;
            --info: #3b82f6;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            flex: 1;
            width: 100%;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        h1 {
            font-size: 1.75rem;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border-radius: 9999px;
            font-size: 0.875rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text-primary), var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-subtitle {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        
        .grid-split {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        
        @media (max-width: 1024px) {
            .grid-split {
                grid-template-columns: 1fr;
            }
        }
        
        .section-title {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .panel {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.05);
            height: 500px;
            display: flex;
            flex-direction: column;
        }
        
        .activity-list {
            list-style: none;
            overflow-y: auto;
            flex: 1;
        }
        
        .activity-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .activity-item:last-child {
            border-bottom: none;
        }
        
        .activity-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
        }
        
        .activity-icon.success { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .activity-icon.error { background: rgba(239, 68, 68, 0.2); color: var(--error); }
        
        .activity-info {
            display: flex;
            align-items: center;
            flex: 1;
        }
        
        .activity-name {
            font-weight: 500;
        }
        
        .activity-time {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        
        /* Log Console Styles */
        .log-console {
            background: #000000;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.8rem;
            color: #d4d4d4;
            overflow-y: auto;
            flex: 1;
            scroll-behavior: smooth;
        }
        
        .log-entry {
            margin-bottom: 0.25rem;
            line-height: 1.4;
            word-wrap: break-word;
        }
        
        .log-time {
            color: #569cd6;
            margin-right: 0.5rem;
        }
        
        .log-level {
            display: inline-block;
            width: 60px;
            font-weight: bold;
        }
        
        .level-INFO { color: var(--success); }
        .level-WARNING { color: var(--warning); }
        .level-ERROR { color: var(--error); }
        .level-DEBUG { color: var(--text-secondary); }
        
        .log-logger {
            color: #c586c0;
            margin-right: 0.5rem;
        }
        
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }
        
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-card); 
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.1); 
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255,255,255,0.2); 
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Pipeline Dashboard</h1>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span id="connection-status">Connected</span>
            </div>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Files Processed</div>
                <div class="metric-value" id="files-processed">0</div>
                <div class="metric-subtitle">Total successful</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value" id="success-rate">0%</div>
                <div class="metric-subtitle">Processing accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg. Processing Time</div>
                <div class="metric-value" id="avg-time">0s</div>
                <div class="metric-subtitle">Per file</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Tokens</div>
                <div class="metric-value" id="total-tokens">0</div>
                <div class="metric-subtitle">AI usage</div>
            </div>
        </div>
        
        <div class="grid-split">
            <div>
                <h2 class="section-title">Recent Activity</h2>
                <div class="panel">
                    <ul class="activity-list" id="activity-list">
                        <li class="empty-state">No recent activity</li>
                    </ul>
                </div>
            </div>
            
            <div>
                <h2 class="section-title">
                    Live Logs
                    <button onclick="clearLogs()" style="background:none; border:none; color:var(--text-secondary); cursor:pointer; font-size:0.8rem;">Clear</button>
                </h2>
                <div class="panel" style="padding: 0; overflow: hidden;">
                    <div class="log-console" id="log-console">
                        <div class="log-entry"><span class="log-time">System</span> <span class="log-level level-INFO">INFO</span> Waiting for logs...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        const activityList = document.getElementById('activity-list');
        const logConsole = document.getElementById('log-console');
        const activities = [];
        const MAX_LOGS = 500;
        
        ws.onopen = () => {
            document.getElementById('connection-status').textContent = 'Connected';
            fetchMetrics();
            fetchHistory();
        };
        
        ws.onclose = () => {
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.querySelector('.status-dot').style.background = 'var(--error)';
        };
        
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            
            if (message.type === 'file_processed') {
                addActivity(message.data);
            } else if (message.type === 'metrics_update') {
                updateMetrics(message.data);
            } else if (message.type === 'log') {
                addLog(message.data);
            }
        };
        
        async function fetchMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                updateMetrics(data);
            } catch (e) {
                console.error('Failed to fetch metrics:', e);
            }
        }

        async function fetchHistory() {
            try {
                const response = await fetch('/api/history');
                const data = await response.json();
                if (data.items && data.items.length > 0) {
                    activities.push(...data.items);
                    renderActivities();
                }
            } catch (e) {
                console.error('Failed to fetch history:', e);
            }
        }
        
        function updateMetrics(data) {
            document.getElementById('files-processed').textContent = data.files_processed || 0;
            document.getElementById('success-rate').textContent = 
                ((data.success_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('avg-time').textContent = 
                (data.avg_processing_time || 0).toFixed(2) + 's';
            document.getElementById('total-tokens').textContent = 
                (data.total_tokens || 0).toLocaleString();
        }
        
        function addActivity(data) {
            activities.unshift(data);
            if (activities.length > 10) activities.pop();
            renderActivities();
        }
        
        function renderActivities() {
            if (activities.length === 0) {
                activityList.innerHTML = '<li class="empty-state">No recent activity</li>';
                return;
            }
            
            activityList.innerHTML = activities.map(a => `
                <li class="activity-item">
                    <div class="activity-info">
                        <div class="activity-icon ${a.success ? 'success' : 'error'}">
                            ${a.success ? '✓' : '✗'}
                        </div>
                        <span class="activity-name">${a.file_name}</span>
                    </div>
                    <span class="activity-time">${a.processing_time.toFixed(2)}s</span>
                </li>
            `).join('');
        }
        
        function addLog(data) {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            
            // Format log level class (INFO, WARNING, ERROR, DEBUG)
            const levelClass = `level-${data.level}`;
            
            entry.innerHTML = `
                <span class="log-time">${data.timestamp}</span>
                <span class="log-level ${levelClass}">${data.level}</span>
                <span class="log-logger">[${data.logger}]</span>
                <span class="log-message">${escapeHtml(data.message)}</span>
            `;
            
            logConsole.appendChild(entry);
            
            // Auto-scroll to bottom
            logConsole.scrollTop = logConsole.scrollHeight;
            
            // Prune old logs
            if (logConsole.children.length > MAX_LOGS) {
                logConsole.removeChild(logConsole.firstChild);
            }
        }
        
        function clearLogs() {
            logConsole.innerHTML = '';
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Refresh metrics every 30 seconds
        setInterval(fetchMetrics, 30000);
    </script>
</body>
</html>"""
