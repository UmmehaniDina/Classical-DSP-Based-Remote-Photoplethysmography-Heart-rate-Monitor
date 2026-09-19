"""Peak-based BPM estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal
from numpy.typing import ArrayLike, NDArray

from rppg.config import AnalysisConfig


@dataclass(frozen=True)
class PeakEstimate:
    """Peak-interval estimate plus the waveform polarity that produced it."""

    bpm: float
    peaks: NDArray[np.int_]
    polarity: str


def _estimate_single_polarity(
    filtered: ArrayLike,
    uniform_times: ArrayLike,
    fps: float,
    band_hz: ArrayLike,
    cfg: AnalysisConfig,
) -> tuple[float, NDArray[np.int_]]:
    filtered = np.asarray(filtered, dtype=np.float64).reshape(-1)
    uniform_times = np.asarray(uniform_times, dtype=np.float64).reshape(-1)
    empty_peaks = np.array([], dtype=int)

    if filtered.size == 0 or uniform_times.size == 0 or filtered.size != uniform_times.size:
        return np.nan, empty_peaks
    if len(band_hz) != 2 or not np.all(np.isfinite(band_hz)):
        return np.nan, empty_peaks
    if not np.isfinite(fps) or fps <= 0:
        return np.nan, empty_peaks

    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    min_bpm = max(cfg.min_accepted_bpm, band_hz[0] * 60.0)
    max_bpm = min(cfg.max_accepted_bpm, band_hz[1] * 60.0)
    if max_bpm <= min_bpm:
        return np.nan, empty_peaks

    min_distance_samples = max(1, int(np.floor(fps * 60.0 / max_bpm)))
    std_val = float(np.std(filtered))
    if not np.isfinite(std_val):
        std_val = 0.0
    prominence = max(cfg.peak_prominence_std_fraction * std_val, cfg.min_peak_prominence)

    peaks, _ = signal.find_peaks(filtered, distance=min_distance_samples, prominence=prominence)
    if len(peaks) < 3:
        return np.nan, peaks

    intervals = np.diff(uniform_times[peaks])
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) == 0:
        return np.nan, peaks

    interval_bpm = 60.0 / intervals
    valid = (interval_bpm >= min_bpm) & (interval_bpm <= max_bpm) & np.isfinite(interval_bpm)
    interval_bpm = interval_bpm[valid]
    if len(interval_bpm) < 2:
        return np.nan, peaks

    median_bpm = float(np.median(interval_bpm))
    mad = float(np.median(np.abs(interval_bpm - median_bpm))) + 1e-9
    interval_bpm = interval_bpm[np.abs(interval_bpm - median_bpm) <= 3.5 * mad]
    if len(interval_bpm) == 0:
        return np.nan, peaks
    return float(np.median(interval_bpm)), peaks


def estimate_peaks_bpm(
    filtered: ArrayLike,
    uniform_times: ArrayLike,
    fps: float,
    band_hz: ArrayLike,
    cfg: AnalysisConfig,
    *,
    allow_inverted: bool = True,
) -> tuple[float, NDArray[np.int_]]:
    """Estimate BPM from peak intervals, optionally checking both polarities.

    Args:
        filtered: Filtered rPPG candidate samples.
        uniform_times: Sample timestamps in seconds aligned with filtered.
        fps: Effective sample rate in samples/s.
        band_hz: Accepted frequency band in hertz.
        cfg: Analysis settings for BPM limits and peak prominence.
        allow_inverted: If true, evaluate troughs by peak-finding on -signal.

    Returns:
        Median peak-interval BPM and integer peak indexes. BPM is NaN when the
        peak estimate is unavailable.
    """
    estimate = estimate_peaks_bpm_detailed(
        filtered,
        uniform_times,
        fps,
        band_hz,
        cfg,
        allow_inverted=allow_inverted,
    )
    return estimate.bpm, estimate.peaks


def estimate_peaks_bpm_detailed(
    filtered: ArrayLike,
    uniform_times: ArrayLike,
    fps: float,
    band_hz: ArrayLike,
    cfg: AnalysisConfig,
    *,
    allow_inverted: bool = True,
) -> PeakEstimate:
    """Estimate BPM and retain the selected normal/inverted peak polarity."""
    bpm_pos, peaks_pos = _estimate_single_polarity(filtered, uniform_times, fps, band_hz, cfg)
    if not (allow_inverted and cfg.peak_detect_both_polarities):
        return PeakEstimate(bpm_pos, peaks_pos, "normal")

    bpm_neg, peaks_neg = _estimate_single_polarity(-np.asarray(filtered, dtype=np.float64), uniform_times, fps, band_hz, cfg)
    if np.isfinite(bpm_pos) and not np.isfinite(bpm_neg):
        return PeakEstimate(bpm_pos, peaks_pos, "normal")
    if np.isfinite(bpm_neg) and not np.isfinite(bpm_pos):
        return PeakEstimate(bpm_neg, peaks_neg, "inverted")
    if np.isfinite(bpm_pos) and np.isfinite(bpm_neg):
        if len(peaks_neg) > len(peaks_pos):
            return PeakEstimate(bpm_neg, peaks_neg, "inverted")
    return PeakEstimate(bpm_pos, peaks_pos, "normal")
