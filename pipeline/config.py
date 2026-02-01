"""Configuration management for the Data Processing Pipeline.

This module defines Pydantic-based configuration models for all aspects of the
pipeline, including directory management, processing parameters, LLM generator
settings, and logging.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DirectoriesConfig(BaseModel):
    """Configuration for pipeline-related directories.

    Attributes:
        data: Path to the input data directory.
        output: Path where processing results are saved.
        logs: Path for storing application logs and metrics.
    """
    data: str = "data"
    output: str = "output"
    logs: str = "logs"


class ProcessingConfig(BaseModel):
    """Settings for file processing and concurrency.

    Attributes:
        supported_extensions: List of file extensions the pipeline will process.
        max_file_size_mb: Maximum file size allowed for processing.
        concurrent_workers: Number of parallel processing workers.
        retry_attempts: Number of times to retry failed processing jobs.
        retry_delay_seconds: Seconds to wait between retries.
    """
    supported_extensions: list[str] = Field(default_factory=lambda: [".txt", ".md", ".json", ".csv", ".pdf", ".xlsx"])
    max_file_size_mb: int = 50
    concurrent_workers: int = 4
    retry_attempts: int = 3
    retry_delay_seconds: int = 5


class GeneratorConfig(BaseModel):
    """Settings for the AI instruction generator.

    Attributes:
        provider: The LLM provider to use (e.g., 'gemini', 'ollama').
        model: The specific model name. Use 'auto' for default.
        temperature: Sampling temperature for the model.
        max_tokens: Maximum tokens to generate per file.
        ollama_host: URL of the Ollama server (only used if provider is 'ollama').
        instruction_template: Jinja2-style template for the generator prompt.
    """
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
    """Configuration for application logging.

    Attributes:
        level: Standard log level (INFO, DEBUG, etc.).
        format: Log format (e.g., 'json', 'simple').
        rotation: Settings for log file rotation.
    """
    level: str = "INFO"
    format: str = "json"
    
    class RotationConfig(BaseModel):
        """Settings for log file rotation."""
        max_size_mb: int = 10
        backup_count: int = 5
    
    rotation: RotationConfig = Field(default_factory=RotationConfig)


class WatcherConfig(BaseModel):
    """Settings for the file system watcher.

    Attributes:
        debounce_seconds: Time to wait after a file change before processing.
        recursive: Whether to watch subdirectories.
        ignore_patterns: Filename patterns to ignore.
    """
    debounce_seconds: float = 1.0
    recursive: bool = True
    ignore_patterns: list[str] = Field(default_factory=lambda: ["*.tmp", "*.swp", ".*"])


class MemoryConfig(BaseModel):
    """Configuration for vector memory (Pinecone).

    Attributes:
        pinecone_api_key: Optional API key for Pinecone authentication.
        pinecone_environment: Pinecone cloud environment (e.g., 'us-east-1-gcp').
        pinecone_index_name: Name of the index to store embeddings in.
    """
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str | None = None


class PipelineConfig(BaseModel):
    """Main container for all pipeline configuration settings.

    Provides methods to load from YAML, handle environment overrides, and
    resolve absolute directory paths.
    """
    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    
    _base_path: Path = Path(".")
    
    @classmethod
    def load(cls, config_path: str | None = None, base_path: Path | None = None) -> PipelineConfig:
        """Load configuration from a file or environment.

        Resolution order:
        1. Explicitly passed `config_path`.
        2. Environment variable `PIPELINE_CONFIG`.
        3. Default `config.yaml` in the project root.

        Args:
            config_path: Custom path to config file.
            base_path: Root path to resolve relative directories against.

        Returns:
            A validated PipelineConfig instance.
        """
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
