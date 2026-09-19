"""Timestamp normalization and frame-window quality checks."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rppg.config import AnalysisConfig
from rppg.exceptions import LowQualitySignalError, WarmupError


@dataclass(frozen=True)
class FrameWindowMetrics:
    """Acquisition and detection metrics calculated from all window frames."""

    effective_fps: float
    detection_availability: float
    sample_coverage: float
    max_frame_gap_sec: float
    max_sample_gap_sec: float


class CaptureClock:
    """Choose progressing capture timestamps, falling back to monotonic time."""

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._last_timestamp_sec = -np.inf
        self._last_capture_timestamp_sec: float | None = None

    def next_timestamp(self, capture_timestamp_ms: float | None) -> tuple[float, str]:
        """Return a strictly increasing timestamp and its source label."""
        monotonic_timestamp = time.monotonic() - self._started_at
        capture_timestamp = (
            float(capture_timestamp_ms) / 1000.0 if capture_timestamp_ms is not None else np.nan
        )
        use_capture_timestamp = (
            np.isfinite(capture_timestamp)
            and capture_timestamp > 0.0
            and capture_timestamp > self._last_timestamp_sec
            and (
                self._last_capture_timestamp_sec is None
                or capture_timestamp > self._last_capture_timestamp_sec
            )
        )
        if use_capture_timestamp:
            timestamp = capture_timestamp
            source = "capture"
            self._last_capture_timestamp_sec = capture_timestamp
        else:
            timestamp = monotonic_timestamp
            source = "monotonic"

        timestamp = max(timestamp, self._last_timestamp_sec + 1e-6)
        self._last_timestamp_sec = timestamp
        return timestamp, source


def _strictly_increasing_times(timestamps: ArrayLike) -> NDArray[np.float64]:
    """Return finite, sorted, de-duplicated timestamps."""
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return np.unique(values)


def frame_window_metrics(
    frame_times: ArrayLike,
    face_detected: Sequence[bool],
    sample_times: ArrayLike,
) -> FrameWindowMetrics:
    """Measure frame cadence, detection availability, and accepted-sample coverage."""
    frames = _strictly_increasing_times(frame_times)
    samples = _strictly_increasing_times(sample_times)
    if len(frames) < 2:
        raise WarmupError("Collecting frame timing data...")
    if len(face_detected) != len(frame_times):
        raise LowQualitySignalError("Frame timestamps and face detection records do not match.")

    duration = float(frames[-1] - frames[0])
    if duration <= 0.0:
        raise WarmupError("Collecting frame timing data...")
    effective_fps = float((len(frames) - 1) / duration)
    frame_gaps = np.diff(frames)
    max_frame_gap_sec = float(np.max(frame_gaps))
    expected_samples = max(1.0, duration * effective_fps)
    sample_coverage = float(min(1.0, len(samples) / expected_samples))
    detection_availability = float(np.count_nonzero(face_detected) / len(face_detected))
    max_sample_gap_sec = float(np.max(np.diff(samples))) if len(samples) >= 2 else float("inf")
    return FrameWindowMetrics(
        effective_fps=effective_fps,
        detection_availability=detection_availability,
        sample_coverage=sample_coverage,
        max_frame_gap_sec=max_frame_gap_sec,
        max_sample_gap_sec=max_sample_gap_sec,
    )


def validate_frame_window(
    frame_times: ArrayLike,
    face_detected: Sequence[bool],
    sample_times: ArrayLike,
    current_time: float,
    cfg: AnalysisConfig,
) -> FrameWindowMetrics:
    """Reject windows with stale, gapped, or insufficiently covered data."""
    metrics = frame_window_metrics(frame_times, face_detected, sample_times)
    if metrics.effective_fps < cfg.min_effective_fps:
        raise LowQualitySignalError(f"All-frame effective FPS {metrics.effective_fps:.2f} is too low.")
    frame_interval = 1.0 / metrics.effective_fps
    if metrics.max_frame_gap_sec > cfg.max_timestamp_gap_factor * frame_interval:
        raise LowQualitySignalError(
            f"Frame timestamp gap {metrics.max_frame_gap_sec:.3f}s exceeds the allowed limit."
        )
    if metrics.sample_coverage < cfg.min_sample_coverage_fraction:
        raise LowQualitySignalError(
            f"Sample coverage {metrics.sample_coverage:.2%} is below the required minimum."
        )
    if metrics.max_sample_gap_sec > cfg.max_timestamp_gap_factor * frame_interval:
        raise LowQualitySignalError(
            f"Accepted-sample gap {metrics.max_sample_gap_sec:.3f}s exceeds the allowed limit."
        )
    samples = _strictly_increasing_times(sample_times)
    if len(samples) == 0:
        raise WarmupError("No accepted samples available for analysis.")
    age = float(current_time - samples[-1])
    if age > cfg.max_sample_age_sec:
        raise LowQualitySignalError(f"Latest accepted sample is stale by {age:.3f}s.")
    return metrics
