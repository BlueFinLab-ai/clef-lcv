"""Compatibility import for the unified Clef service (flash profile)."""
import importlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("CLEF_PROFILE", "flash")
service = importlib.import_module("clef_service.app")
if service.PROFILE_NAME != "flash":
    raise RuntimeError("One model profile per process; start a separate service for the other model")
sys.modules[__name__] = service
