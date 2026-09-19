"""Launcher for the Classical rPPG Monitor Desktop Application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rppg.gui import launch_gui


def main() -> int:
    """Launch the GUI application."""
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
