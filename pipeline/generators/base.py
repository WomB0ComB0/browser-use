from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.extractors.base import ExtractedContent


@dataclass
class GeneratedInstructions:
    """Container for generated instructions."""
    
    # Core content
    instructions: str
    title: str
    
    # Source information
    source_file: Path
    source_type: str
    
    # Generation metadata
    generated_at: datetime = field(default_factory=datetime.now)
    model_used: str | None = None
    tokens_used: int | None = None
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# {self.title}",
            "",
            f"> Generated from: `{self.source_file.name}`",
            f"> File type: {self.source_type}",
            f"> Generated at: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            self.instructions,
        ]
        return "\n".join(lines)
    
    def save(self, output_path: Path) -> Path:
        """Save instructions to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        
        return output_path


class BaseGenerator(ABC):
    """Abstract base class for instruction generators."""
    
    @abstractmethod
    async def generate(self, content: ExtractedContent) -> GeneratedInstructions:
        """Generate instructions from extracted content.
        
        Args:
            content: The extracted content to generate instructions from.
            
        Returns:
            GeneratedInstructions containing the AI-generated instructions.
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        pass
