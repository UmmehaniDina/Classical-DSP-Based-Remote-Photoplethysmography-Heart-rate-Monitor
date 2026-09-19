"""UBFC-rPPG dataset-folder support and BPM evaluation helpers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from rppg.exceptions import ConfigurationError

UBFC_VIDEO_NAME = "vid.avi"
UBFC_GROUND_TRUTH_NAME = "ground_truth.txt"


def resolve_ubfc_dataset(dataset_dir: str | Path) -> tuple[Path, Path]:
    """Return the UBFC video and annotation paths from a dataset folder."""
    folder = Path(dataset_dir).expanduser()
    video_path = folder / UBFC_VIDEO_NAME
    ground_truth_path = folder / UBFC_GROUND_TRUTH_NAME
    if not folder.is_dir():
        raise ConfigurationError(f"UBFC dataset folder does not exist: {folder}")
    if not video_path.is_file():
        raise ConfigurationError(f"UBFC dataset folder is missing {UBFC_VIDEO_NAME}: {folder}")
    if not ground_truth_path.is_file():
        raise ConfigurationError(f"UBFC dataset folder is missing {UBFC_GROUND_TRUTH_NAME}: {folder}")
    return video_path, ground_truth_path


def is_ubfc_dataset_folder(source: str | Path) -> bool:
    """Return whether *source* is a directory with the UBFC required files."""
    folder = Path(source).expanduser()
    return folder.is_dir() and (folder / UBFC_VIDEO_NAME).is_file() and (folder / UBFC_GROUND_TRUTH_NAME).is_file()


def _numeric_rows(path: Path) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", line)]
        finite_values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
        if finite_values.size:
            rows.append(finite_values)
    return rows


def _timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Normalize UBFC timestamps, which may be stored in seconds or milliseconds."""
    relative = timestamps - timestamps[0]
    if relative.size < 2:
        return relative
    median_step = float(np.median(np.diff(relative)))
    if not np.isfinite(median_step) or median_step <= 0.0:
        raise ConfigurationError("UBFC ground_truth.txt timestamps must be strictly increasing.")

    # UBFC releases commonly store seconds, but some exported copies use ms/us.
    # Select the unit that produces a plausible video-frame interval near 30 FPS.
    candidates = (1.0, 1e-3, 1e-6)
    scale = min(candidates, key=lambda item: abs(np.log10((median_step * item) / (1.0 / 30.0))))
    result = relative * scale
    if np.any(np.diff(result) <= 0.0):
        raise ConfigurationError("UBFC ground_truth.txt timestamps must be strictly increasing.")
    return result


def load_ubfc_reference(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load UBFC heart-rate samples and timestamps from ``ground_truth.txt``.

    The official UBFC-rPPG layout has three numeric rows: pulse waveform,
    heart-rate BPM, and timestamps. Only the BPM and timestamp rows are used.
    """
    _, ground_truth_path = resolve_ubfc_dataset(dataset_dir)
    rows = _numeric_rows(ground_truth_path)
    if len(rows) < 3:
        raise ConfigurationError(
            "UBFC ground_truth.txt must contain waveform, BPM, and timestamp rows (the official three-row format)."
        )
    bpm, timestamps = rows[1], rows[2]
    count = min(bpm.size, timestamps.size)
    if count < 2:
        raise ConfigurationError("UBFC ground_truth.txt needs at least two BPM/timestamp samples.")
    bpm = bpm[:count]
    timestamps = timestamps[:count]
    valid = np.isfinite(bpm) & np.isfinite(timestamps)
    bpm, timestamps = bpm[valid], timestamps[valid]
    if bpm.size < 2:
        raise ConfigurationError("UBFC ground_truth.txt has too few finite BPM/timestamp samples.")
    order = np.argsort(timestamps)
    return _timestamps_to_seconds(timestamps[order]), bpm[order]


def _read_estimates(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    bpm_values: list[float] = []
    with (session_dir / "csv" / "bpm_estimates.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                time_sec = float(row["time_sec"])
                bpm = float(row["smoothed_bpm"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(time_sec) and np.isfinite(bpm):
                times.append(time_sec)
                bpm_values.append(bpm)
    return np.asarray(times, dtype=np.float64), np.asarray(bpm_values, dtype=np.float64)


def _read_frame_times(session_dir: Path) -> np.ndarray:
    """Read decoded-frame timestamps for the frame-level error display."""
    times: list[float] = []
    path = session_dir / "csv" / "frame_events.csv"
    if not path.exists():
        return np.asarray(times, dtype=np.float64)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                time_sec = float(row["time_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(time_sec):
                times.append(time_sec)
    return np.asarray(times, dtype=np.float64)


def evaluate_ubfc_session(session_dir: str | Path, dataset_dir: str | Path) -> dict[str, Any]:
    """Write aligned UBFC BPM errors and return aggregate error metrics."""
    session = Path(session_dir)
    reference_times, reference_bpm = load_ubfc_reference(dataset_dir)
    estimate_times, estimate_bpm = _read_estimates(session)
    in_range = (estimate_times >= reference_times[0]) & (estimate_times <= reference_times[-1])
    aligned_times = estimate_times[in_range]
    aligned_estimates = estimate_bpm[in_range]
    aligned_reference = np.interp(aligned_times, reference_times, reference_bpm)
    errors = aligned_estimates - aligned_reference

    comparison_path = session / "csv" / "ubfc_bpm_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_sec", "estimated_bpm", "ground_truth_bpm", "error_bpm", "absolute_error_bpm"])
        for time_sec, estimated, reference, error in zip(aligned_times, aligned_estimates, aligned_reference, errors):
            writer.writerow([time_sec, estimated, reference, error, abs(error)])

    # BPM is produced once per analysis update, not once per video frame. For a
    # clearer frame-by-frame plot, linearly interpolate only between valid BPM
    # updates; no value is invented before the first or after the last update.
    frame_times = _read_frame_times(session)
    frame_start = max(reference_times[0], aligned_times[0]) if aligned_times.size else np.inf
    frame_end = min(reference_times[-1], aligned_times[-1]) if aligned_times.size else -np.inf
    valid_frames = (frame_times >= frame_start) & (frame_times <= frame_end)
    comparison_frame_times = frame_times[valid_frames]
    if comparison_frame_times.size:
        frame_estimated = np.interp(comparison_frame_times, aligned_times, aligned_estimates)
        frame_reference = np.interp(comparison_frame_times, reference_times, reference_bpm)
        frame_errors = frame_estimated - frame_reference
    else:
        frame_estimated = frame_reference = frame_errors = np.asarray([], dtype=np.float64)

    frame_error_path = session / "csv" / "ubfc_frame_error.csv"
    with frame_error_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_sec", "estimated_bpm", "ground_truth_bpm", "error_bpm", "absolute_error_bpm"])
        for time_sec, estimated, reference, error in zip(
            comparison_frame_times, frame_estimated, frame_reference, frame_errors
        ):
            writer.writerow([time_sec, estimated, reference, error, abs(error)])

    report: dict[str, Any] = {
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "video": str(resolve_ubfc_dataset(dataset_dir)[0]),
        "ground_truth": str(resolve_ubfc_dataset(dataset_dir)[1]),
        "comparison_csv": str(comparison_path),
        "frame_error_csv": str(frame_error_path),
        "final_result_graph": str(session / "figures" / "ubfc_final_results.png"),
        "paired_estimate_count": int(errors.size),
        "frame_error_count": int(frame_errors.size),
    }
    if errors.size:
        report.update(
            {
                "mean_estimated_bpm": float(np.mean(aligned_estimates)),
                "mean_ground_truth_bpm": float(np.mean(aligned_reference)),
                "mean_absolute_bpm_error": float(np.mean(np.abs(errors))),
                "root_mean_squared_bpm_error": float(np.sqrt(np.mean(np.square(errors)))),
                "mean_bpm_bias": float(np.mean(errors)),
            }
        )
    report_path = session / "metadata" / "ubfc_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
