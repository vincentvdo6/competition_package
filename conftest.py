"""Pytest configuration for Wunderfund Predictorium project."""

import sys
import os

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
