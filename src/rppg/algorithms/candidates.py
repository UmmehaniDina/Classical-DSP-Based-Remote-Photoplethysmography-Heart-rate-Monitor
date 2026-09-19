"""Candidate rPPG signal generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rppg.config import AnalysisConfig
from rppg.dsp.preprocessing import zscore_safe
from rppg.dsp.spectral import spectral_quality_and_bpm

logger = logging.getLogger("rppg_monitor")


@dataclass(frozen=True)
class CandidateSignal:
    """One rPPG candidate signal plus method metadata."""

    name: str
    signal: NDArray[np.float64]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_inverted_peaks: bool = True


def _valid_rgb_matrix(rgb_norm: NDArray[np.float64]) -> bool:
    return rgb_norm.ndim == 2 and rgb_norm.shape[1] == 3 and rgb_norm.size > 0


def make_green_candidate(rgb_norm: NDArray[np.float64]) -> CandidateSignal:
    """Return the green-channel baseline in the common candidate interface."""
    return CandidateSignal(
        name="Green",
        signal=zscore_safe(rgb_norm[:, 1]),
        method="green",
        metadata={"channel": "G", "polarity": "green-positive"},
    )


def _windowed_projection(
    rgb: NDArray[np.float64],
    fps: float,
    cfg: AnalysisConfig,
    *,
    method: str,
    temporal_normalize: bool = False,
) -> tuple[NDArray[np.float64], dict[str, float | int | str]]:
    """Apply CHROM or POS with overlapping local alpha estimates.

    A whole-window alpha can be distorted when lighting or motion changes
    during the analysis period. This uses 50% overlapping local windows and
    averages the resulting projections, preserving one output sample per input
    sample while avoiding hard chunk boundaries.
    """
    if not _valid_rgb_matrix(rgb) or not np.isfinite(fps) or fps <= 0.0:
        return np.zeros(0, dtype=np.float64), {"reason": "invalid_input"}
    n_samples = len(rgb)
    window_samples = min(n_samples, max(8, int(round(cfg.candidate_projection_window_sec * fps))))
    step_samples = max(1, window_samples // 2)
    starts = list(range(0, max(n_samples - window_samples + 1, 1), step_samples))
    final_start = max(0, n_samples - window_samples)
    if final_start not in starts:
        starts.append(final_start)

    projected = np.zeros(n_samples, dtype=np.float64)
    weights = np.zeros(n_samples, dtype=np.float64)
    alphas: list[float] = []
    for start in sorted(set(starts)):
        stop = min(n_samples, start + window_samples)
        segment = rgb[start:stop]
        if temporal_normalize:
            channel_means = np.mean(segment, axis=0)
            if not np.all(np.isfinite(channel_means)) or np.any(np.abs(channel_means) <= 1e-9):
                continue
            segment = (segment / channel_means) - 1.0
        r, g, b = segment[:, 0], segment[:, 1], segment[:, 2]
        if method == "chrom":
            first = (3.0 * r) - (2.0 * g)
            second = (1.5 * r) + g - (1.5 * b)
            sign = -1.0
        else:
            first = g - b
            second = (-2.0 * r) + g + b
            sign = 1.0
        second_std = float(np.std(second))
        alpha = float(np.std(first) / second_std) if second_std > 1e-12 else 0.0
        candidate = first + sign * alpha * second
        # Taper overlapping projections so their joins do not create sharp
        # discontinuities that can leak into the pulse band. The positive
        # floor keeps the first/last samples represented in the final signal.
        taper = 0.1 + (0.9 * np.hanning(len(candidate)))
        projected[start:stop] += candidate * taper
        weights[start:stop] += taper
        alphas.append(alpha)

    signal = np.divide(projected, weights, out=np.zeros_like(projected), where=weights > 0.0)
    return signal, {
        "projection": "overlapping_local_alpha",
        "projection_window_sec": float(cfg.candidate_projection_window_sec),
        "projection_window_samples": window_samples,
        "projection_window_count": len(alphas),
        "alpha_median": float(np.median(alphas)) if alphas else float("nan"),
        "input_normalization": "window_channel_mean" if temporal_normalize else "preprocessed_fallback",
        "projection_taper": "hann_with_0.1_floor",
    }


def make_chrom_candidate(
    rgb: NDArray[np.float64],
    fps: float,
    cfg: AnalysisConfig,
    *,
    temporal_normalize: bool = False,
) -> CandidateSignal:
    """Return a CHROM baseline with overlapping local projection weights."""
    chrom, metadata = _windowed_projection(rgb, fps, cfg, method="chrom", temporal_normalize=temporal_normalize)
    return CandidateSignal(
        name="CHROM",
        signal=zscore_safe(chrom),
        method="chrom",
        metadata={**metadata, "formula": "X=3R-2G; Y=1.5R+G-1.5B; S=X-alphaY"},
    )


def make_pos_candidate(
    rgb: NDArray[np.float64],
    fps: float,
    cfg: AnalysisConfig,
    *,
    temporal_normalize: bool = False,
) -> CandidateSignal:
    """Return a POS baseline with overlapping local projection weights."""
    pos, metadata = _windowed_projection(rgb, fps, cfg, method="pos", temporal_normalize=temporal_normalize)
    return CandidateSignal(
        name="POS",
        signal=zscore_safe(pos),
        method="pos",
        metadata={**metadata, "formula": "S1=G-B; S2=-2R+G+B; S=S1+alphaS2"},
    )


def _align_component_sign(
    component: NDArray[np.float64],
    reference: NDArray[np.float64],
    cfg: AnalysisConfig,
) -> tuple[NDArray[np.float64], float]:
    finite = np.isfinite(component) & np.isfinite(reference)
    if int(np.count_nonzero(finite)) < cfg.pca_sign_min_samples:
        return component, float("nan")
    corr = float(np.corrcoef(component[finite], reference[finite])[0, 1])
    if not np.isfinite(corr) or abs(corr) < cfg.pca_sign_min_abs_correlation:
        return component, corr
    if corr < 0.0:
        return -component, corr
    return component, corr


def make_pca_signal(
    rgb_norm: NDArray[np.float64],
    fps: float,
    search_band: ArrayLike,
    cfg: AnalysisConfig,
) -> CandidateSignal:
    """Generate the best PCA component by explicit quality criteria.

    PCA is kept as a baseline, not a claimed blind-source separator. Component
    selection combines spectral quality with explained variance and rejects
    near-constant components. This can still select periodic motion or lighting
    artifacts, so CHROM/POS/Green are evaluated alongside it.

    Args:
        rgb_norm: Nx3 detrended and normalized RGB samples.
        fps: Effective sample rate in samples/s.
        search_band: Low/high frequency band in hertz.
        cfg: Analysis settings.
    """
    empty = np.zeros(0, dtype=np.float64)
    if not _valid_rgb_matrix(rgb_norm) or not np.any(np.isfinite(rgb_norm)):
        return CandidateSignal("PCA", empty, "pca", {"reason": "empty_input"})

    centered = rgb_norm - np.mean(rgb_norm, axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("SVD failed in PCA candidate generation; falling back to green channel.")
        return CandidateSignal("PCA", zscore_safe(rgb_norm[:, 1]), "pca", {"fallback": "green"})

    components = centered @ vt.T
    if components.size == 0:
        return CandidateSignal("PCA", zscore_safe(rgb_norm[:, 1]), "pca", {"fallback": "green"})

    variance = singular_values**2
    total_variance = float(np.sum(variance))
    variance_ratio = variance / total_variance if total_variance > 0.0 else np.zeros_like(variance)

    best_score = -np.inf
    best_component = components[:, 0]
    best_index = 0
    best_snr = -np.inf
    for i in range(components.shape[1]):
        component = components[:, i]
        component_std = float(np.std(component))
        if not np.isfinite(component_std) or component_std < cfg.pca_min_component_std:
            continue
        score, _, _, _ = spectral_quality_and_bpm(components[:, i], fps, search_band, cfg)
        quality_score = float(score) * np.sqrt(float(variance_ratio[i]) + 1e-12)
        if np.isfinite(quality_score) and quality_score > best_score:
            best_score = quality_score
            best_component = component
            best_index = i
            best_snr = float(score)

    if not np.any(np.isfinite(best_component)):
        return CandidateSignal("PCA", zscore_safe(rgb_norm[:, 1]), "pca", {"fallback": "green"})

    best_component, sign_corr = _align_component_sign(best_component, rgb_norm[:, 1], cfg)
    return CandidateSignal(
        name="PCA",
        signal=zscore_safe(best_component),
        method="pca",
        metadata={
            "component_index": best_index,
            "selection_score": best_score,
            "spectral_snr": best_snr,
            "variance_ratio": float(variance_ratio[best_index]) if len(variance_ratio) else float("nan"),
            "green_sign_correlation": sign_corr,
            "limitation": "PCA may select periodic motion or illumination artifacts.",
        },
    )


def make_ica_candidate(
    rgb_norm: NDArray[np.float64],
    fps: float,
    search_band: ArrayLike,
    cfg: AnalysisConfig,
) -> CandidateSignal | None:
    """Return an optional ICA/BSS candidate when sklearn is installed/enabled."""
    if not cfg.enable_ica_candidate:
        return None
    try:
        from sklearn.decomposition import FastICA
    except Exception:
        logger.warning("ICA candidate requested but scikit-learn is unavailable.")
        return None
    if not _valid_rgb_matrix(rgb_norm):
        return CandidateSignal("ICA", np.zeros(0, dtype=np.float64), "ica", {"reason": "empty_input"})

    ica = FastICA(n_components=3, random_state=0, whiten="unit-variance", max_iter=300)
    try:
        components = ica.fit_transform(np.nan_to_num(rgb_norm, nan=0.0, posinf=0.0, neginf=0.0))
    except Exception:
        logger.warning("ICA candidate generation failed.", exc_info=True)
        return None

    best_score = -np.inf
    best_component = components[:, 0]
    best_index = 0
    for i in range(components.shape[1]):
        score, _, _, _ = spectral_quality_and_bpm(components[:, i], fps, search_band, cfg)
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_component = components[:, i]
            best_index = i
    best_component, sign_corr = _align_component_sign(best_component, rgb_norm[:, 1], cfg)
    return CandidateSignal(
        name="ICA",
        signal=zscore_safe(best_component),
        method="ica",
        metadata={"component_index": best_index, "spectral_snr": best_score, "green_sign_correlation": sign_corr},
    )


def generate_candidate_signals(
    rgb_norm: NDArray[np.float64],
    fps: float,
    search_band: ArrayLike,
    cfg: AnalysisConfig,
    *,
    rgb_for_projection: NDArray[np.float64] | None = None,
) -> tuple[CandidateSignal, ...]:
    """Return classical rPPG candidates in the common candidate interface.

    GREEN and PCA use the detrended/z-scored matrix. CHROM/POS use raw RGB
    normalized by each local projection window when ``rgb_for_projection`` is
    supplied; this preserves their color-projection assumptions.
    """
    if not _valid_rgb_matrix(rgb_norm):
        return tuple()
    projection_rgb = rgb_norm if rgb_for_projection is None else np.asarray(rgb_for_projection, dtype=np.float64)
    if projection_rgb.shape != rgb_norm.shape:
        return tuple()
    temporal_normalize = rgb_for_projection is not None
    candidates: list[CandidateSignal] = [
        make_green_candidate(rgb_norm),
        make_pca_signal(rgb_norm, fps, search_band, cfg),
        make_chrom_candidate(projection_rgb, fps, cfg, temporal_normalize=temporal_normalize),
        make_pos_candidate(projection_rgb, fps, cfg, temporal_normalize=temporal_normalize),
    ]
    ica = make_ica_candidate(rgb_norm, fps, search_band, cfg)
    if ica is not None:
        candidates.append(ica)
    return tuple(candidates)


def candidate_rppg_signals(
    rgb_norm: NDArray[np.float64],
    fps: float,
    search_band: ArrayLike,
    cfg: AnalysisConfig,
) -> dict[str, NDArray[np.float64]]:
    """Return legacy name-to-signal mapping for tests or external callers."""
    return {candidate.name: candidate.signal for candidate in generate_candidate_signals(rgb_norm, fps, search_band, cfg)}
