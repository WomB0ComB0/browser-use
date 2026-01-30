from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DirectoriesConfig(BaseModel):
    """Directory configuration."""
    data: str = "data"
    output: str = "output"
    logs: str = "logs"


class ProcessingConfig(BaseModel):
    """File processing configuration."""
    supported_extensions: list[str] = Field(default_factory=lambda: [".txt", ".md", ".json", ".csv"])
    max_file_size_mb: int = 50
    concurrent_workers: int = 4
    retry_attempts: int = 3
    retry_delay_seconds: int = 5


class GeneratorConfig(BaseModel):
    """AI generator configuration."""
    provider: str = "gemini"  # gemini, ollama
    model: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 4096
    ollama_host: str = "http://localhost:11434"
    instruction_template: str = """Analyze the following data and generate clear, actionable instructions:

## Data Type: {file_type}
## Content Summary: {summary}

Generate instructions that explain:
1. What this data represents
2. How it should be used
3. Key patterns or insights
4. Recommended actions"""


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    
    class RotationConfig(BaseModel):
        max_size_mb: int = 10
        backup_count: int = 5
    
    rotation: RotationConfig = Field(default_factory=RotationConfig)


class WatcherConfig(BaseModel):
    """File watcher configuration."""
    debounce_seconds: float = 1.0
    recursive: bool = True
    ignore_patterns: list[str] = Field(default_factory=lambda: ["*.tmp", "*.swp", ".*"])


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    
    _base_path: Path = Path(".")
    
    @classmethod
    def load(cls, config_path: str | None = None, base_path: Path | None = None) -> PipelineConfig:
        """Load configuration from YAML file with environment variable overrides."""
        if base_path is None:
            base_path = Path(".")
        
        if config_path is None:
            config_path = os.environ.get("PIPELINE_CONFIG", "config.yaml")
        
        config_file = base_path / config_path
        
        if config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        
        # Apply environment variable overrides
        if "PIPELINE_DATA_DIR" in os.environ:
            data.setdefault("directories", {})["data"] = os.environ["PIPELINE_DATA_DIR"]
        if "PIPELINE_OUTPUT_DIR" in os.environ:
            data.setdefault("directories", {})["output"] = os.environ["PIPELINE_OUTPUT_DIR"]
        if "PIPELINE_LOG_LEVEL" in os.environ:
            data.setdefault("logging", {})["level"] = os.environ["PIPELINE_LOG_LEVEL"]
        if "PIPELINE_PROVIDER" in os.environ:
            data.setdefault("generator", {})["provider"] = os.environ["PIPELINE_PROVIDER"]
        if "PIPELINE_MODEL" in os.environ:
            data.setdefault("generator", {})["model"] = os.environ["PIPELINE_MODEL"]
        
        config = cls.model_validate(data)
        config._base_path = base_path
        return config
    
    def get_data_dir(self) -> Path:
        """Get absolute path to data directory."""
        return self._base_path / self.directories.data
    
    def get_output_dir(self) -> Path:
        """Get absolute path to output directory."""
        return self._base_path / self.directories.output
    
    def get_logs_dir(self) -> Path:
        """Get absolute path to logs directory."""
        return self._base_path / self.directories.logs
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        self.get_data_dir().mkdir(parents=True, exist_ok=True)
        self.get_output_dir().mkdir(parents=True, exist_ok=True)
        self.get_logs_dir().mkdir(parents=True, exist_ok=True)
