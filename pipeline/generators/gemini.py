"""Gemini-based instruction generator."""

import os
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI

from pipeline.config import PipelineConfig
from pipeline.extractors.base import ExtractedContent
from pipeline.generators.base import BaseGenerator, GeneratedInstructions
from pipeline.utils.logging import get_logger


class GeminiGenerator(BaseGenerator):
    """Generate instructions using Google Gemini."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Initialize the model
        self.llm = ChatGoogleGenerativeAI(
            model=config.generator.model,
            google_api_key=api_key,
            temperature=config.generator.temperature,
            max_tokens=config.generator.max_tokens,
        )
    
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        return self.config.generator.model
    
    async def generate(self, content: ExtractedContent) -> GeneratedInstructions:
        """Generate instructions from extracted content."""
        self.logger.info(f"Generating instructions for: {content.file_name}")
        
        # Build the prompt
        prompt = self._build_prompt(content)
        
        # Generate response
        try:
            response = await self.llm.ainvoke(prompt)
            instructions = response.content
            
            # Extract token usage if available
            tokens_used = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = response.usage_metadata.get("total_tokens")
            
        except Exception as e:
            self.logger.error(f"Error generating instructions: {e}")
            instructions = self._create_fallback_instructions(content, str(e))
            tokens_used = None
        
        # Create title from filename
        title = self._create_title(content)
        
        return GeneratedInstructions(
            instructions=instructions,
            title=title,
            source_file=content.file_path,
            source_type=content.file_type,
            generated_at=datetime.now(),
            model_used=self.get_model_name(),
            tokens_used=tokens_used,
            metadata={
                "file_size": content.file_size_bytes,
                "summary_length": len(content.summary),
            },
        )
    
    def _build_prompt(self, content: ExtractedContent) -> str:
        """Build the prompt for instruction generation."""
        # Get template from config
        template = self.config.generator.instruction_template
        
        # Format structure info
        structure_info = ""
        if content.structure:
            structure_info = self._format_structure(content.structure)
        
        # Build context about the file
        context = f"""
## File Information
- **Name**: {content.file_name}
- **Type**: {content.file_type}
- **Size**: {content.file_size_bytes:,} bytes
- **Modified**: {content.modified_time.strftime('%Y-%m-%d %H:%M:%S')}

## Content Summary
{content.summary}

{structure_info}

## Full Content
```
{self._truncate_content(content.content)}
```
"""
        
        # Apply template
        prompt = template.format(
            file_type=content.file_type,
            summary=content.summary,
        )
        
        return f"{prompt}\n\n{context}"
    
    def _format_structure(self, structure: dict) -> str:
        """Format structure information for the prompt."""
        if not structure:
            return ""
        
        lines = ["## Structure Analysis"]
        
        # Handle different structure types
        if "headers" in structure:  # Markdown
            if structure["headers"]:
                lines.append("### Document Headers")
                for h in structure["headers"][:10]:
                    indent = "  " * (h["level"] - 1)
                    lines.append(f"{indent}- {h['text']}")
        
        if "columns" in structure:  # CSV
            lines.append("### Column Information")
            for col in structure["columns"][:10]:
                lines.append(f"- **{col['name']}** ({col['type']}): {col['unique_count']} unique values")
        
        if "keys" in structure:  # JSON
            lines.append("### Object Keys")
            lines.append(f"- Keys: {', '.join(structure['keys'][:10])}")
        
        return "\n".join(lines)
    
    def _truncate_content(self, content: str, max_chars: int = 10000) -> str:
        """Truncate content if too long."""
        if len(content) <= max_chars:
            return content
        
        return content[:max_chars] + f"\n\n... [Truncated, {len(content) - max_chars:,} more characters]"
    
    def _create_title(self, content: ExtractedContent) -> str:
        """Create a title from the filename."""
        name = content.file_path.stem
        # Convert snake_case or kebab-case to Title Case
        title = name.replace("_", " ").replace("-", " ").title()
        return f"Instructions for {title}"
    
    def _create_fallback_instructions(self, content: ExtractedContent, error: str) -> str:
        """Create fallback instructions when AI generation fails."""
        return f"""## Automatic Generation Failed

An error occurred while generating instructions: {error}

### File Summary

- **File**: {content.file_name}
- **Type**: {content.file_type}
- **Size**: {content.file_size_bytes:,} bytes

### Content Preview

{content.summary}

---

*Please review the source file manually and contact support if this issue persists.*
"""
