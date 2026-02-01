#!/usr/bin/env python3
"""Convenience entry point for the pipeline CLI.

Maps the global execution context to the internal typer application defined 
in `pipeline.cli`.
"""
"""Main entry point for the Enterprise Data Pipeline."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.cli import main

if __name__ == "__main__":
    main()
