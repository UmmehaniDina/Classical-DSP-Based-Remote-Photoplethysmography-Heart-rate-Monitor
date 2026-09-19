import os
import sys
from pathlib import Path

tcl_dir = os.path.join(sys.base_prefix, "tcl")
if os.path.isdir(tcl_dir):
    tcl_lib = os.path.join(tcl_dir, "tcl8.6")
    tk_lib = os.path.join(tcl_dir, "tk8.6")
    if os.path.isdir(tcl_lib):
        os.environ.setdefault("TCL_LIBRARY", tcl_lib)
    if os.path.isdir(tk_lib):
        os.environ.setdefault("TK_LIBRARY", tk_lib)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
