"""Welch spectral estimation and spectral quality metrics."""

from __future__ import annotations

import numpy as np
import scipy.signal as signal
from numpy.typing import ArrayLike, NDArray

from rppg.config import AnalysisConfig
from rppg.dsp.filtering import clamp_band_to_nyquist


def welch_spectrum(x: ArrayLike, fps: float, cfg: AnalysisConfig) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Welch spectrum for a signal.

    Args:
        x: One-dimensional signal samples.
        fps: Effective sample rate in samples/s.
        cfg: Analysis settings for Welch window sizing.

    Returns:
        Frequency bins in hertz and spectrum power values.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(fps) or fps <= 0:
        return np.array([]), np.array([])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    resolution_samples = int(round(fps / cfg.welch_target_resolution_hz))
    time_window_samples = int(round(cfg.welch_window_sec * fps))
    requested = max(cfg.welch_min_segment_samples, min(time_window_samples, resolution_samples))
    nperseg = min(len(x), requested)
    if nperseg > len(x):
        nperseg = len(x)
    if nperseg < 2:
        return np.array([]), np.array([])

    return signal.welch(
        x - np.mean(x),
        fs=fps,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="spectrum",
    )


def refine_peak_frequency(
    freqs: ArrayLike,
    power: ArrayLike,
    peak_index: int,
) -> float:
    """Refine a discrete spectral peak with local parabolic interpolation."""
    freqs = np.asarray(freqs, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    if peak_index <= 0 or peak_index >= len(power) - 1 or len(freqs) != len(power):
        return float(freqs[peak_index])
    step = float(freqs[1] - freqs[0])
    if not np.isfinite(step) or step <= 0.0:
        return float(freqs[peak_index])
    y0, y1, y2 = np.log(np.maximum(power[peak_index - 1 : peak_index + 2], 1e-24))
    denom = y0 - (2.0 * y1) + y2
    if abs(float(denom)) <= 1e-12:
        return float(freqs[peak_index])
    offset = 0.5 * (y0 - y2) / denom
    if not np.isfinite(offset) or abs(float(offset)) > 1.0:
        return float(freqs[peak_index])
    return float(freqs[peak_index] + offset * step)


def harmonic_adjusted_frequency(
    freqs: ArrayLike,
    power: ArrayLike,
    peak_freq: float,
    band: ArrayLike,
    cfg: AnalysisConfig,
) -> tuple[float, float]:
    """Reduce octave errors by checking plausible subharmonics/harmonics.

    Returns the adjusted frequency and a quality multiplier. The correction is
    deliberately conservative: a subharmonic only replaces the spectral maximum
    when it has a meaningful fraction of the peak power and lies in-band.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    band = np.asarray(band, dtype=np.float64)
    if len(freqs) == 0 or len(power) == 0 or not np.isfinite(peak_freq):
        return peak_freq, 1.0

    peak_power = float(np.interp(peak_freq, freqs, power))
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        return peak_freq, 1.0

    def local_power(target_hz: float, tolerance_hz: float) -> float:
        local = np.abs(freqs - target_hz) <= tolerance_hz
        if np.any(local):
            return float(np.max(power[local]))
        return float(np.interp(target_hz, freqs, power))

    adjusted = float(peak_freq)
    for divisor in (2.0, 3.0):
        sub = peak_freq / divisor
        if sub < band[0] or sub > band[1]:
            continue
        sub_power = local_power(sub, cfg.subharmonic_tolerance_hz)
        if sub_power >= cfg.harmonic_power_fraction * peak_power:
            adjusted = float(sub)
            break

    quality_multiplier = 1.0
    for multiplier in (2.0, 3.0):
        harmonic = adjusted * multiplier
        if harmonic < freqs[0] or harmonic > freqs[-1]:
            continue
        harmonic_power = local_power(harmonic, cfg.harmonic_tolerance_hz)
        if harmonic_power >= cfg.harmonic_power_fraction * peak_power:
            quality_multiplier = cfg.harmonic_quality_bonus
            break

    return adjusted, quality_multiplier


def spectral_edge_penalty_multiplier(peak_freq: float, band: ArrayLike, cfg: AnalysisConfig) -> float:
    """Return a symmetric penalty for spectral maxima near either band edge."""
    try:
        low, high = (float(band[0]), float(band[1]))
    except (IndexError, TypeError, ValueError):
        return 1.0
    if not np.isfinite(peak_freq) or high <= low or cfg.spectral_edge_penalty_hz <= 0.0:
        return 1.0
    edge_distance = min(peak_freq - low, high - peak_freq)
    return cfg.spectral_edge_penalty if edge_distance <= cfg.spectral_edge_penalty_hz else 1.0


def spectral_quality_and_bpm(
    x: ArrayLike,
    fps: float,
    search_band: ArrayLike,
    cfg: AnalysisConfig,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Estimate dominant spectral BPM and an SNR-like quality score."""
    freqs, power = welch_spectrum(x, fps, cfg)
    if len(freqs) == 0 or len(power) == 0:
        return -np.inf, np.nan, freqs, power
    if not np.any(np.isfinite(power)):
        return -np.inf, np.nan, freqs, power

    power = np.where(np.isfinite(power), power, 0.0)
    band = clamp_band_to_nyquist(search_band, fps, cfg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return -np.inf, np.nan, freqs, power

    band_freqs = freqs[mask]
    band_power = power[mask]
    if len(band_power) == 0 or not np.any(np.isfinite(band_power)):
        return -np.inf, np.nan, freqs, power

    peak_idx = int(np.argmax(band_power))
    global_peak_idx = int(np.flatnonzero(mask)[peak_idx])
    peak_freq = float(band_freqs[peak_idx])
    if cfg.spectral_refine_peak:
        peak_freq = refine_peak_frequency(freqs, power, global_peak_idx)
    if not np.isfinite(peak_freq):
        return -np.inf, np.nan, freqs, power
    # A parabolic fit uses neighboring bins, which can lie outside the search
    # mask when the maximum is at a band edge. Never let that extrapolate the
    # reported estimate beyond the configured physiological range.
    peak_freq = float(np.clip(peak_freq, band[0], band[1]))

    exclude = np.abs(band_freqs - peak_freq) <= cfg.spectral_peak_exclusion_hz
    if np.any(~exclude):
        noise_power = float(np.median(band_power[~exclude]))
    else:
        noise_power = float(np.median(band_power))
    if not np.isfinite(noise_power):
        noise_power = 0.0

    snr = float(band_power[peak_idx]) / (noise_power + 1e-12)
    if not np.isfinite(snr):
        snr = -np.inf
    peak_freq, harmonic_bonus = harmonic_adjusted_frequency(freqs, power, peak_freq, band, cfg)
    edge_penalty = spectral_edge_penalty_multiplier(peak_freq, band, cfg)
    return snr * edge_penalty * harmonic_bonus, peak_freq * 60.0, freqs, power
