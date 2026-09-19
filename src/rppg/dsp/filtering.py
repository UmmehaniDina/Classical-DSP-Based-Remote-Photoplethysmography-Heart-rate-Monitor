"""Frequency-band helpers and filters."""

from __future__ import annotations

import numpy as np
import scipy.signal as signal
from numpy.typing import ArrayLike, NDArray

from rppg.config import AnalysisConfig
from rppg.exceptions import AnalysisError, LowQualitySignalError


def clamp_band_to_nyquist(band: ArrayLike, fps: float, cfg: AnalysisConfig) -> list[float]:
    """Clamp a frequency band to the valid range for the effective FPS.

    Args:
        band: Two-value low/high frequency band in hertz.
        fps: Effective sample rate in frames/s.
        cfg: Analysis settings containing frequency safety margins.

    Raises:
        AnalysisError: If the band values are malformed or non-finite.
        LowQualitySignalError: If fps is too low for the requested band.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise AnalysisError("Invalid FPS for band clamping.")
    try:
        low_in = float(band[0])
        high_in = float(band[1])
    except (TypeError, ValueError, IndexError):
        raise AnalysisError("Invalid search band.")
    if not np.isfinite(low_in) or not np.isfinite(high_in):
        raise AnalysisError("Non-finite search band.")

    nyq = fps / 2.0
    low = max(cfg.min_band_low_hz, low_in)
    high = min(high_in, cfg.nyquist_high_fraction * nyq)
    if high <= low:
        raise LowQualitySignalError(f"FPS {fps:.2f} is too low for band {band}.")
    return [low, high]


def make_adaptive_band(last_valid_bpm: float | None, fps: float, cfg: AnalysisConfig) -> list[float]:
    """Return the global band or a previous-BPM-centered adaptive band."""
    global_band = clamp_band_to_nyquist(cfg.global_search_band_hz, fps, cfg)
    if last_valid_bpm is None or not np.isfinite(last_valid_bpm):
        return global_band

    center = last_valid_bpm / 60.0
    low = max(global_band[0], center - cfg.adaptive_half_bandwidth_hz)
    high = min(global_band[1], center + cfg.adaptive_half_bandwidth_hz)
    if high - low < cfg.min_adaptive_bandwidth_hz:
        expand = (cfg.min_adaptive_bandwidth_hz - (high - low)) / 2.0
        low = max(global_band[0], low - expand)
        high = min(global_band[1], high + expand)
    if high <= low:
        return global_band
    return [low, high]


def bandpass_filter(data: ArrayLike, fps: float, band: ArrayLike, cfg: AnalysisConfig) -> NDArray[np.float64]:
    """Apply a zero-phase Butterworth bandpass filter.

    The default third-order Butterworth is intentionally conservative for
    short rPPG windows: it gives a useful pulse-band rolloff while limiting
    ringing from the low-frequency cutoff. Edge handling is explicit through
    scipy.signal.filtfilt padding settings instead of relying on hidden
    defaults.

    Raises:
        WarmupError-compatible LowQualitySignalError: If the input window is
        too short for stable filtering.
        AnalysisError: If the frequency band is malformed.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        return data
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.size < max(16, cfg.butterworth_order * 6):
        raise LowQualitySignalError("Bandpass input is too short.")

    band = clamp_band_to_nyquist(band, fps, cfg)
    nyq = fps / 2.0
    b, a = signal.butter(
        cfg.butterworth_order,
        [band[0] / nyq, band[1] / nyq],
        btype="bandpass",
    )
    padlen = min(data.size - 1, 3 * max(len(a), len(b)))
    return signal.filtfilt(b, a, data, padtype=cfg.filter_padtype, padlen=padlen)
