"""
Pytest configuration file.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent  # Go up one level from tests to project root
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Ensure the path exists
if not src_path.exists():
    raise ImportError(f"Source directory not found: {src_path}")

# Set PYTHONPATH environment variable as well
current_pythonpath = os.environ.get('PYTHONPATH', '')
if str(src_path) not in current_pythonpath:
    new_pythonpath = f"{src_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(src_path)
    os.environ['PYTHONPATH'] = new_pythonpath
