"""Command-line interface for the rPPG monitor."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from rppg.config import AppConfig, default_config, validate_config, with_overrides


VIDEO_FILE_TYPES = [
    (
        "Video files",
        "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm *.mpeg *.mpg",
    ),
    ("All files", "*.*"),
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the compatibility launcher and package CLI."""
    parser = argparse.ArgumentParser(description="Classical MediaPipe-based rPPG monitor.")
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index, UBFC dataset folder, video path, or stream URI. Skips the launch-time source selector.",
    )
    parser.add_argument(
        "--roi-mode",
        default=None,
        choices=["forehead", "upperCheeks", "multi"],
        help="Landmark ROI selection mode.",
    )
    parser.add_argument(
        "--skin-mask-strategy",
        choices=["fixed", "adaptive_chroma", "temporal_adaptive"],
        default=None,
        help="Skin-mask strategy; fixed is the default, while adaptive modes are comparison-only research heuristics.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV preview and Matplotlib dashboard.")
    parser.add_argument("--debug-roi", action="store_true", help="Overlay ROI polygons and skin-coverage diagnostics.")
    parser.add_argument(
        "--analysis-window-seconds",
        type=float,
        default=None,
        help="Analysis-window duration in seconds; use a fixed value when comparing methods.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=None,
        help="Minimum accepted-sample duration before analysis; must be shorter than the analysis window.",
    )
    parser.add_argument(
        "--candidate-projection-window-seconds",
        type=float,
        default=None,
        help="Local CHROM/POS projection window duration in seconds.",
    )
    parser.add_argument(
        "--global-band-probe-interval-updates",
        type=int,
        default=None,
        help="Run a global frequency reset/probe every N analysis updates (default: 10).",
    )
    parser.add_argument(
        "--calibrate-seconds",
        type=float,
        default=None,
        help="Record a stationary face/skin-quality baseline and exit without BPM estimation.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run without live displays and emit an acquisition summary when the source ends.",
    )
    parser.add_argument("--output-dir", default=None, help="Root directory for timestamped session outputs.")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console and file logging level.",
    )
    parser.add_argument(
        "--camera-backend",
        default=None,
        choices=["auto", "dshow", "msmf", "v4l2", "avfoundation", "gstreamer"],
        help="OpenCV capture backend preference.",
    )
    parser.add_argument("--manual-exposure", type=float, default=None, help="Requested manual camera exposure value.")
    parser.add_argument("--manual-gain", type=float, default=None, help="Requested manual camera gain value.")
    parser.add_argument(
        "--manual-white-balance", type=float, default=None, help="Requested manual white-balance temperature/value."
    )
    parser.add_argument("--manual-focus", type=float, default=None, help="Requested manual camera focus value.")
    return parser


def choose_saved_video_file(input_fn: Callable[[str], str] = input) -> str | None:
    """Open a native video-file picker, with a terminal-path fallback.

    The fallback keeps saved-video replay usable on systems without a desktop
    session or a working Tk installation.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        path = input_fn("Enter the full path to the saved video (blank to cancel): ").strip()
        return path or None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            selected = filedialog.askopenfilename(title="Choose a saved video", filetypes=VIDEO_FILE_TYPES)
        finally:
            root.destroy()
        return selected or None
    except (OSError, RuntimeError, tk.TclError):
        path = input_fn("Enter the full path to the saved video (blank to cancel): ").strip()
        return path or None


def choose_ubfc_dataset_folder(input_fn: Callable[[str], str] = input) -> str | None:
    """Open a folder picker for a UBFC subject folder, with terminal fallback."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        path = input_fn("Enter the UBFC dataset folder (blank to cancel): ").strip()
        return path or None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(title="Choose UBFC dataset folder (vid.avi + ground_truth.txt)")
        finally:
            root.destroy()
        return selected or None
    except (OSError, RuntimeError, tk.TclError):
        path = input_fn("Enter the UBFC dataset folder (blank to cancel): ").strip()
        return path or None


def choose_input_source(
    input_fn: Callable[[str], str] = input,
    file_chooser: Callable[[Callable[[str], str]], str | None] = choose_ubfc_dataset_folder,
) -> str | None:
    """Ask whether to read from the live camera or a UBFC dataset folder."""
    while True:
        try:
            choice = input_fn("Choose reading mode: [1] Live camera  [2] UBFC dataset folder: ").strip().lower()
        except EOFError:
            # Non-interactive launchers retain the long-standing camera-default behavior.
            return "0"

        if choice in {"1", "live", "live camera", "camera", "l"}:
            return "0"
        if choice in {"2", "ubfc", "dataset", "dataset folder", "folder", "d", "saved", "video"}:
            return file_chooser(input_fn)
        print("Please enter 1 for live camera or 2 for a UBFC dataset folder.")


def config_from_args(argv: list[str] | None = None) -> AppConfig:
    """Parse CLI arguments and return a validated application configuration."""
    args = build_parser().parse_args(argv)
    return config_from_namespace(args)


def config_from_namespace(args: argparse.Namespace) -> AppConfig:
    """Build a validated application configuration from parsed CLI arguments."""
    cfg = with_overrides(
        default_config(),
        source=args.source,
        roi_mode=args.roi_mode,
        skin_mask_strategy=args.skin_mask_strategy,
        headless=args.headless,
        debug_roi=args.debug_roi,
        calibration_duration_sec=args.calibrate_seconds,
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        log_level=args.log_level,
        camera_backend=args.camera_backend,
        manual_exposure=args.manual_exposure,
        manual_gain=args.manual_gain,
        manual_white_balance=args.manual_white_balance,
        manual_focus=args.manual_focus,
        analysis_window_sec=args.analysis_window_seconds,
        min_warmup_sec=args.warmup_seconds,
        candidate_projection_window_sec=args.candidate_projection_window_seconds,
        global_band_probe_interval_updates=args.global_band_probe_interval_updates,
    )
    validate_config(cfg)
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Run the rPPG monitor from CLI arguments."""
    from rppg.app.runner import run

    args = build_parser().parse_args(argv)
    if args.source is None:
        selected_source = choose_input_source()
        if selected_source is None:
            print("No UBFC dataset folder was selected; exiting.")
            return 0
        args.source = selected_source

    return run(config_from_namespace(args))
