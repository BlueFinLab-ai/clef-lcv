"""Compatibility import for the shared compact embedding implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clef_service.compact_quant import *
