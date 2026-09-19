"""Window-level rPPG analysis and estimate fusion."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d

from rppg.algorithms.candidates import CandidateSignal, generate_candidate_signals
from rppg.config import AnalysisConfig
from rppg.dsp.filtering import bandpass_filter, clamp_band_to_nyquist, make_adaptive_band
from rppg.dsp.peaks import estimate_peaks_bpm_detailed
from rppg.dsp.preprocessing import preprocess_rgb
from rppg.dsp.spectral import spectral_quality_and_bpm
from rppg.exceptions import AnalysisError, LowQualitySignalError, WarmupError
from rppg.results import AnalysisResult

logger = logging.getLogger("rppg_monitor")


def multi_feature_sqi(
    spectral_score: float,
    bpm_fft: float,
    bpm_peaks: float,
    peak_count: int,
    cfg: AnalysisConfig,
    context: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float | int]]:
    """Combine spectral prominence with independent peak-agreement evidence.

    A dominant spectrum alone can be caused by motion or lighting. Peak
    agreement is intentionally a soft multiplier: unavailable peaks reduce
    confidence but do not manufacture a rejection from a short otherwise clean
    window. The returned feature map is stored with each accepted result.
    """
    if not np.isfinite(spectral_score):
        return float("-inf"), {"spectral_score": float(spectral_score), "peak_count": peak_count}
    if np.isfinite(bpm_fft) and np.isfinite(bpm_peaks):
        difference = abs(bpm_fft - bpm_peaks)
        agreement = float(np.exp(-difference / cfg.peak_match_tolerance_bpm))
    else:
        difference = float("nan")
        agreement = 0.5
    context = context or {}

    def bounded(name: str, default: float) -> float:
        value = float(context.get(name, default))
        return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else default

    # These are soft quality factors. Hard timing, motion, and illumination
    # safety limits remain enforced in their dedicated acquisition gates.
    sample_coverage = bounded("sample_coverage", 1.0)
    detection_availability = bounded("detection_availability", 1.0)
    mask_coverage = bounded("mask_coverage", 1.0)
    motion_quality = bounded("motion_quality", 1.0)
    illumination_quality = bounded("illumination_quality", 1.0)
    acquisition_quality = float(
        np.mean([sample_coverage, detection_availability, mask_coverage, motion_quality, illumination_quality])
    )
    quality = float(spectral_score) * (0.5 + 0.5 * agreement) * acquisition_quality
    return quality, {
        "spectral_score": float(spectral_score),
        "peak_agreement": agreement,
        "peak_bpm_difference": float(difference),
        "peak_count": int(peak_count),
        "sample_coverage": sample_coverage,
        "detection_availability": detection_availability,
        "mask_coverage": mask_coverage,
        "motion_quality": motion_quality,
        "illumination_quality": illumination_quality,
        "acquisition_quality": acquisition_quality,
    }


def fuse_estimates(
    bpm_fft: float,
    bpm_peaks: float,
    quality: float,
    previous_bpm: float | None,
    cfg: AnalysisConfig,
) -> float:
    """Fuse spectral and peak estimates with quality and jump guards.

    Returns NaN when the estimate fails quality, physiological-range, or
    frame-to-frame jump checks.
    """
    if not np.isfinite(bpm_fft) or not np.isfinite(quality) or quality < cfg.min_spectral_snr:
        return np.nan
    bpm, _ = fuse_estimates_detailed(bpm_fft, bpm_peaks, quality, previous_bpm, cfg)
    return bpm


def fuse_estimates_detailed(
    bpm_fft: float,
    bpm_peaks: float,
    quality: float,
    previous_bpm: float | None,
    cfg: AnalysisConfig,
) -> tuple[float, dict[str, float]]:
    """Fuse estimates and expose reliability-derived weights for audit logs.

    Spectral BPM remains the anchor. Peak BPM is given up to its configured
    maximum weight only when it agrees with the spectrum; disagreement reduces
    its influence continuously rather than creating a hard threshold jump.
    """
    configured_total = float(cfg.peak_fusion_fft_weight) + float(cfg.peak_fusion_peak_weight)
    maximum_peak_weight = float(cfg.peak_fusion_peak_weight) / configured_total if configured_total > 0.0 else 0.0
    metadata = {
        "fusion_fft_weight": 1.0,
        "fusion_peak_weight": 0.0,
        "fusion_peak_agreement": 0.0,
        "fusion_peak_weight_cap": maximum_peak_weight,
    }
    if not np.isfinite(bpm_fft) or not np.isfinite(quality) or quality < cfg.min_spectral_snr:
        return np.nan, metadata

    bpm = float(bpm_fft)
    if np.isfinite(bpm_peaks):
        difference = abs(float(bpm_peaks) - float(bpm_fft))
        agreement = float(np.exp(-difference / cfg.peak_match_tolerance_bpm))
        peak_weight = maximum_peak_weight * agreement
        fft_weight = 1.0 - peak_weight
        bpm = (fft_weight * float(bpm_fft)) + (peak_weight * float(bpm_peaks))
        metadata = {
            "fusion_fft_weight": fft_weight,
            "fusion_peak_weight": peak_weight,
            "fusion_peak_agreement": agreement,
            "fusion_peak_weight_cap": maximum_peak_weight,
        }
    if previous_bpm is not None and np.isfinite(previous_bpm):
        if abs(bpm - previous_bpm) > cfg.max_bpm_jump_per_update:
            return np.nan, metadata
    if cfg.min_accepted_bpm <= bpm <= cfg.max_accepted_bpm:
        return bpm, metadata
    return np.nan, metadata


def construct_uniform_signal(
    sample_times: ArrayLike,
    rgb_samples: ArrayLike,
    cfg: AnalysisConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Interpolate irregular RGB samples onto a uniform time base.

    Args:
        sample_times: One-dimensional sample timestamps in seconds.
        rgb_samples: Nx3 RGB samples in R, G, B order.
        cfg: Analysis settings for minimum sample counts and FPS.

    Raises:
        WarmupError: If more samples or stable timing are needed.
        LowQualitySignalError: If the signal/timing quality is unusable.
        AnalysisError: If input shapes are invalid.
    """
    times = np.asarray(sample_times, dtype=np.float64)
    rgb = np.asarray(rgb_samples, dtype=np.float64)
    if times.size == 0 or rgb.size == 0:
        raise WarmupError("No samples available for analysis.")
    if times.ndim != 1:
        raise AnalysisError("Invalid timestamp shape.")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise AnalysisError("Invalid RGB sample shape.")
    if times.shape[0] != rgb.shape[0]:
        raise AnalysisError("Timestamp and RGB sample counts do not match.")

    finite_time = np.isfinite(times)
    times = times[finite_time]
    rgb = rgb[finite_time]
    if len(times) == 0:
        raise WarmupError("No finite timestamps available for analysis.")

    finite_rows = np.any(np.isfinite(rgb), axis=1)
    times = times[finite_rows]
    rgb = rgb[finite_rows]
    if len(times) < cfg.min_interpolation_samples:
        raise WarmupError("Collecting stable timestamps...")

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    rgb = rgb[order]
    unique_times, inverse = np.unique(times, return_inverse=True)
    if len(unique_times) != len(times):
        aggregated_rgb = np.full((len(unique_times), 3), np.nan, dtype=np.float64)
        for channel in range(3):
            finite = np.isfinite(rgb[:, channel])
            sums = np.zeros(len(unique_times), dtype=np.float64)
            counts = np.zeros(len(unique_times), dtype=np.int64)
            np.add.at(sums, inverse[finite], rgb[finite, channel])
            np.add.at(counts, inverse[finite], 1)
            valid = counts > 0
            aggregated_rgb[valid, channel] = sums[valid] / counts[valid]
        times = unique_times
        rgb = aggregated_rgb

    if np.any(np.diff(times) <= 0.0):
        raise AnalysisError("Timestamps must be strictly increasing after aggregation.")

    dt = np.diff(times)
    dt = dt[dt > 0]
    if len(dt) < cfg.min_interpolation_samples:
        raise WarmupError("Collecting stable timestamps...")
    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        raise LowQualitySignalError("Invalid frame timing.")

    fps = 1.0 / median_dt
    if fps < cfg.min_effective_fps:
        raise LowQualitySignalError(f"Effective FPS {fps:.2f} is too low. Lower camera resolution.")

    max_gap = float(np.max(dt))
    if max_gap > cfg.max_interpolation_gap_factor * median_dt:
        raise LowQualitySignalError(
            f"Sample timestamp gap {max_gap:.3f}s exceeds the interpolation limit."
        )

    uniform_count = int(np.floor((times[-1] - times[0]) / median_dt)) + 1
    uniform_times = times[0] + np.arange(uniform_count, dtype=np.float64) * median_dt
    if len(uniform_times) < cfg.min_uniform_samples:
        raise WarmupError("Not enough uniform samples yet.")

    uniform_rgb = np.zeros((len(uniform_times), 3), dtype=np.float64)
    for channel in range(3):
        finite = np.isfinite(rgb[:, channel])
        if int(np.count_nonzero(finite)) < cfg.min_interpolation_samples:
            raise WarmupError("Insufficient finite RGB samples for interpolation.")
        finite_times = times[finite]
        finite_values = rgb[finite, channel]
        uniq_times, uniq_index = np.unique(finite_times, return_index=True)
        if len(uniq_times) < cfg.min_interpolation_samples:
            raise WarmupError("Insufficient unique finite timestamps for interpolation.")
        interpolator = interp1d(
            uniq_times,
            finite_values[uniq_index],
            kind="linear",
            bounds_error=True,
        )
        uniform_rgb[:, channel] = interpolator(uniform_times)

    if not np.any(np.isfinite(uniform_rgb)):
        raise LowQualitySignalError("Interpolated RGB window is empty or non-finite.")
    return uniform_times, uniform_rgb, fps


def _uniform_rgb(
    sample_times: ArrayLike,
    rgb_samples: ArrayLike,
    cfg: AnalysisConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Backward-compatible alias for tests and earlier internal callers."""
    return construct_uniform_signal(sample_times, rgb_samples, cfg)


def illumination_instability(rgb_samples: ArrayLike) -> float:
    """Return a robust normalized frame-to-frame brightness-change metric."""
    rgb = np.asarray(rgb_samples, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) < 2:
        return float("nan")
    brightness = np.mean(rgb, axis=1)
    reference = float(np.median(np.abs(brightness)))
    if not np.isfinite(reference) or reference <= np.finfo(np.float64).eps:
        return float("inf")
    return float(np.percentile(np.abs(np.diff(brightness)), 95) / reference)


def _candidate_search_bands(
    previous_bpm: float | None,
    fps: float,
    cfg: AnalysisConfig,
    analysis_sequence: int | None,
) -> list[tuple[list[float], str]]:
    """Return adaptive plus periodically scheduled global search bands.

    The global probe runs for the first analysis and then at a configured
    interval. It prevents a tracker from remaining locked to a stale local
    band without doing the more expensive global search on every update.
    """
    adaptive_band = make_adaptive_band(previous_bpm, fps, cfg)
    bands: list[tuple[list[float], str]] = [(adaptive_band, "adaptive")]
    should_probe_global = (
        cfg.global_band_probe_enabled
        and (
            analysis_sequence is None
            or analysis_sequence <= 0
            or analysis_sequence % cfg.global_band_probe_interval_updates == 0
        )
    )
    if should_probe_global:
        global_band = clamp_band_to_nyquist(cfg.global_search_band_hz, fps, cfg)
        if not np.allclose(global_band, adaptive_band, rtol=0.0, atol=1e-9):
            bands.append((global_band, "global_periodic"))
    return bands


def _analyze_candidate(
    candidate: CandidateSignal,
    uniform_times: NDArray[np.float64],
    fps: float,
    band: ArrayLike,
    previous_bpm: float | None,
    cfg: AnalysisConfig,
    illumination: float,
    search_mode: str,
    quality_context: Mapping[str, float] | None,
) -> AnalysisResult | None:
    signal = np.asarray(candidate.signal, dtype=np.float64).reshape(-1)
    finite_count = int(np.count_nonzero(np.isfinite(signal)))
    if signal.size == 0 or finite_count < cfg.min_uniform_samples:
        logger.debug("Candidate %s rejected before filtering: empty/insufficient finite samples.", candidate.name)
        return None

    filtered = bandpass_filter(signal, fps, band, cfg)
    if filtered.size == 0 or int(np.count_nonzero(np.isfinite(filtered))) < cfg.min_uniform_samples:
        logger.debug("Candidate %s rejected after filtering: empty/insufficient finite samples.", candidate.name)
        return None
    analysis_times = uniform_times
    if cfg.filter_transient_trim_sec > 0.0:
        trim = int(round(cfg.filter_transient_trim_sec * fps))
        if trim > 0:
            if filtered.size <= (2 * trim) + cfg.min_uniform_samples:
                logger.debug("Candidate %s rejected: filter transient trim leaves too few samples.", candidate.name)
                return None
            filtered = filtered[trim:-trim]
            analysis_times = uniform_times[trim:-trim]

    spectral_score, bpm_fft, freqs, power = spectral_quality_and_bpm(filtered, fps, band, cfg)
    peak_estimate = estimate_peaks_bpm_detailed(
        filtered,
        analysis_times,
        fps,
        band,
        cfg,
        allow_inverted=candidate.allow_inverted_peaks,
    )
    bpm_peaks = peak_estimate.bpm
    peaks = peak_estimate.peaks
    quality, quality_features = multi_feature_sqi(
        spectral_score,
        bpm_fft,
        bpm_peaks,
        len(peaks),
        cfg,
        context=quality_context,
    )
    fused, fusion_features = fuse_estimates_detailed(bpm_fft, bpm_peaks, quality, previous_bpm, cfg)
    return AnalysisResult(
        name=candidate.name,
        quality=float(quality),
        bpm_fft=float(bpm_fft),
        bpm_peaks=float(bpm_peaks),
        fused_bpm=float(fused),
        filtered=np.asarray(filtered, dtype=np.float64),
        peaks=np.asarray(peaks, dtype=int),
        freqs=np.asarray(freqs, dtype=np.float64),
        power=np.asarray(power, dtype=np.float64),
        uniform_times=np.asarray(analysis_times, dtype=np.float64),
        fps=float(fps),
        band=(float(band[0]), float(band[1])),
        illumination_instability=illumination,
        candidate_metadata={
            **candidate.metadata,
            "search_mode": search_mode,
            "peak_polarity": peak_estimate.polarity,
        },
        quality_features={**quality_features, **fusion_features},
    )


def _prefer_candidate(current: AnalysisResult | None, candidate: AnalysisResult, cfg: AnalysisConfig) -> bool:
    if current is None:
        return True
    if not np.isfinite(candidate.fused_bpm):
        return False
    if not np.isfinite(current.fused_bpm):
        return True
    if np.allclose(candidate.band, cfg.global_search_band_hz, rtol=0.0, atol=1e-9):
        return candidate.quality >= current.quality * cfg.global_band_probe_min_quality_ratio
    return candidate.quality > current.quality


def select_best_candidate(
    uniform_times: ArrayLike,
    rgb_norm: ArrayLike,
    fps: float,
    previous_bpm: float | None,
    cfg: AnalysisConfig,
    *,
    illumination: float = float("nan"),
    analysis_sequence: int | None = None,
    rgb_for_projection: ArrayLike | None = None,
    quality_context: Mapping[str, float] | None = None,
) -> AnalysisResult:
    """Run candidate construction, filtering, spectra, peaks, and selection.

    ``analysis_sequence`` is a zero-based update count used to schedule the
    periodic global frequency probe. Omit it for one-off/offline analysis.
    """
    uniform_times = np.asarray(uniform_times, dtype=np.float64)
    rgb_norm = np.asarray(rgb_norm, dtype=np.float64)
    if rgb_norm.ndim != 2 or rgb_norm.shape[1] != 3 or len(rgb_norm) < cfg.min_uniform_samples:
        raise LowQualitySignalError("Candidate input is empty or malformed.")
    projection_rgb = None if rgb_for_projection is None else np.asarray(rgb_for_projection, dtype=np.float64)
    if projection_rgb is not None and projection_rgb.shape != rgb_norm.shape:
        raise AnalysisError("Raw RGB projection input must match the preprocessed RGB shape.")

    best: AnalysisResult | None = None
    for band, search_mode in _candidate_search_bands(previous_bpm, fps, cfg, analysis_sequence):
        try:
            candidates = generate_candidate_signals(
                rgb_norm,
                fps,
                band,
                cfg,
                rgb_for_projection=projection_rgb,
            )
        except AnalysisError:
            raise
        except LowQualitySignalError:
            raise
        except ValueError as exc:
            raise AnalysisError(f"Candidate generation failed: {exc}") from exc

        for candidate in candidates:
            if hasattr(cfg, "extraction_method") and cfg.extraction_method:
                target = cfg.extraction_method.strip().lower()
                if target not in ("all", "select all", "auto", "best"):
                    cand_method = candidate.method.lower()
                    cand_name = candidate.name.lower()
                    if target not in cand_method and target not in cand_name:
                        continue
            try:
                result = _analyze_candidate(
                    candidate,
                    uniform_times,
                    fps,
                    band,
                    previous_bpm,
                    cfg,
                    illumination,
                    search_mode,
                    quality_context,
                )
            except (AnalysisError, LowQualitySignalError):
                logger.debug("Candidate %s failed during processing.", candidate.name, exc_info=True)
                continue
            except ValueError as exc:
                raise AnalysisError(f"Candidate {candidate.name} failed during processing: {exc}") from exc
            if result is not None and _prefer_candidate(best, result, cfg):
                best = result

    if best is None or not np.isfinite(best.fused_bpm):
        raise LowQualitySignalError("Rejected low-quality estimates.")
    return best


def analyze_window(
    sample_times: ArrayLike,
    rgb_samples: ArrayLike,
    previous_bpm: float | None,
    cfg: AnalysisConfig,
    *,
    analysis_sequence: int | None = None,
    quality_context: Mapping[str, float] | None = None,
) -> AnalysisResult:
    """Analyze the current RGB window and return the best accepted estimate.

    Args:
        sample_times: Sample timestamps in seconds.
        rgb_samples: Nx3 RGB samples in R, G, B order.
        previous_bpm: Last accepted BPM used to form the adaptive search band.
        cfg: Analysis settings and thresholds.

    Raises:
        WarmupError: If the window has not collected enough stable samples.
        LowQualitySignalError: If no candidate produces an accepted BPM.
        AnalysisError: If an expected processing failure occurs.
    """
    uniform_times, uniform_rgb, fps = construct_uniform_signal(sample_times, rgb_samples, cfg)
    illumination = illumination_instability(uniform_rgb)
    if not np.isfinite(illumination) or illumination > cfg.max_illumination_instability:
        raise LowQualitySignalError(
            f"Illumination instability {illumination:.3f} exceeds the allowed limit."
        )
    rgb_norm = preprocess_rgb(uniform_rgb, cfg, fps=fps)
    normalized_context = dict(quality_context or {})
    normalized_context["illumination_quality"] = float(
        np.clip(1.0 - (illumination / cfg.max_illumination_instability), 0.0, 1.0)
    )
    return select_best_candidate(
        uniform_times,
        rgb_norm,
        fps,
        previous_bpm,
        cfg,
        illumination=illumination,
        analysis_sequence=analysis_sequence,
        rgb_for_projection=uniform_rgb,
        quality_context=normalized_context,
    )
