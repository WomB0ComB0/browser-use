# Enterprise Data Processing Pipeline

A modular, enterprise-grade data pipeline that monitors folders for new files and automatically generates AI-powered instructions.

## Features

- 📁 **File Watching** - Real-time monitoring of data directories
- 🔄 **Multi-format Support** - Process .txt, .md, .json, .csv files
- 🤖 **AI Instructions** - Automatic instruction generation with Gemini
- 📊 **Structured Logging** - Production-ready logging and metrics
- ⚡ **Async Processing** - High-performance async architecture

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the pipeline
python run_pipeline.py start

# Or process a single file
python run_pipeline.py process data/sample.txt
```

## Configuration

Edit `config.yaml` to customize:
- Input/output directories
- AI model settings
- File type filters
- Logging preferences

## Project Structure

```
pipeline/
├── cli.py           # Command-line interface
├── config.py        # Configuration management
├── watcher.py       # File system watcher
├── processor.py     # Main pipeline processor
├── extractors/      # File type handlers
└── generators/      # Instruction generators
```
