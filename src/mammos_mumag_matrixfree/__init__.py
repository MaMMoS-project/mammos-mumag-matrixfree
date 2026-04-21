"""Matrix-free finite-element micromagnetic software."""

import importlib.metadata
import sys
from pathlib import Path

_loop_bin = f"{sys.executable} {Path(__file__).resolve().parent / 'core' / 'loop.py'}"
_mesh_bin = f"{sys.executable} {Path(__file__).resolve().parent / 'core' / 'mesh.py'}"
__version__ = importlib.metadata.version(__package__)
