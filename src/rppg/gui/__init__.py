"""Modern desktop GUI for the classical rPPG monitor."""

from __future__ import annotations

import os
import sys

# Ensure Windows virtualenvs can locate Tcl/Tk runtime libraries
tcl_dir = os.path.join(sys.base_prefix, "tcl")
if os.path.isdir(tcl_dir):
    tcl_lib = os.path.join(tcl_dir, "tcl8.6")
    tk_lib = os.path.join(tcl_dir, "tk8.6")
    if os.path.isdir(tcl_lib):
        os.environ.setdefault("TCL_LIBRARY", tcl_lib)
    if os.path.isdir(tk_lib):
        os.environ.setdefault("TK_LIBRARY", tk_lib)

from .app_window import RppgAppWindow, launch_gui

__all__ = ["RppgAppWindow", "launch_gui"]
