"""pytest configuration for the get_started_claude_sdk_client package.

Adds the package root to sys.path so that ``util_claude`` can be imported
from any test regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

# Ensure the directory that contains util_claude/ is on the path.
sys.path.insert(0, str(Path(__file__).parent))
