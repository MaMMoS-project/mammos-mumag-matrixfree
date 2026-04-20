"""Test configuration."""

from pathlib import Path
import sys

import pytest

@pytest.fixture(scope="session")
def data_dir():
    """Define data directory."""
    return Path(__file__).resolve().parent / "data"

@pytest.fixture(scope="session")
def loop_bin():
    """Define loop bin."""
    python_bin = sys.executable
    return f"{sys.executable} {Path(__file__).resolve().parent.parent / 'src' / 'loop.py'}"

@pytest.fixture(scope="session")
def mesh_bin():
    """Define mesh bin."""
    python_bin = sys.executable
    return f"{sys.executable} {Path(__file__).resolve().parent.parent / 'src' / 'mesh.py'}"
