#!/usr/bin/env python3
"""Main entry point for the Enterprise Data Pipeline."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.cli import main

if __name__ == "__main__":
    main()
