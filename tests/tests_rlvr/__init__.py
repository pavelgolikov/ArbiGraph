"""Tests for the RLVR interface."""

from pathlib import Path
import sys

# pytest puts tests/ on the path, not the repository root, so point at the
# directory holding arbigraph_rlvr and full_train.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rlvr"))
