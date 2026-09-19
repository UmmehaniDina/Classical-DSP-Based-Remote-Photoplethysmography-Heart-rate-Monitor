"""Detrending and normalization helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from numpy.typing import ArrayLike, NDArray

from rppg.config import AnalysisConfig
from rppg.exceptions import LowQualitySignalError


@lru_cache(maxsize=64)
def _smoothness_priors_factor(n: int, lamb_key: float) -> sparse_linalg.SuperLU:
    """Return a cached sparse factorization for fixed window length/lambda."""
    identity = sparse.eye(n, format="csc")
    diagonals = [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)]
    d2 = sparse.diags(diagonals, [0, 1, 2], shape=(n - 2, n), format="csc")
    system = identity + (float(lamb_key) ** 2) * (d2.T @ d2)
    return sparse_linalg.splu(system)


def smoothness_lambda_for_fps(cfg: AnalysisConfig, fps: float | None) -> float:
    """Scale the smoothness-priors penalty to preserve behavior across FPS.

    The historical value is interpreted at cfg.smoothness_priors_reference_fps.
    Because the second-difference operator is sample-index based, the penalty
    scales with the square of the sampling-rate ratio.
    """
    if fps is None or not np.isfinite(fps) or fps <= 0.0:
        return float(cfg.smoothness_priors_lambda)
    ratio = float(fps) / float(cfg.smoothness_priors_reference_fps)
    return float(cfg.smoothness_priors_lambda) * (ratio**2)


def smoothness_priors_detrend(x: ArrayLike, lamb: float) -> NDArray[np.float64]:
    """Remove slow trend using Tarvainen-style smoothness priors.

    Args:
        x: One-dimensional signal samples.
        lamb: Positive smoothness penalty.

    Returns:
        Detrended signal in the same sample units as x.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if n < 3:
        return x - np.mean(x)

    lamb_key = round(float(lamb), 6)
    trend = _smoothness_priors_factor(n, lamb_key).solve(x)
    return x - trend


def zscore_safe(x: ArrayLike) -> NDArray[np.float64]:
    """Z-score an array, returning zeros for near-constant input."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(x)
    if std < 1e-9:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def preprocess_rgb(rgb: ArrayLike, cfg: AnalysisConfig, fps: float | None = None) -> NDArray[np.float64]:
    """Detrend and z-score an RGB sample matrix.

    Args:
        rgb: Nx3 RGB samples with channels in R, G, B order.
        cfg: Analysis settings, including smoothness-priors lambda.
        fps: Optional effective sample rate for FPS-aware detrending.

    Raises:
        LowQualitySignalError: If RGB input is empty, malformed, or cannot be
        interpolated channel-by-channel.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.size == 0 or rgb.ndim != 2 or rgb.shape[1] != 3:
        raise LowQualitySignalError("Empty or invalid RGB data for preprocessing.")

    out = np.zeros_like(rgb)
    for i in range(3):
        channel = rgb[:, i].copy()
        finite = np.isfinite(channel)
        if not np.any(finite):
            raise LowQualitySignalError(f"RGB channel {i} contains no finite samples.")
        if np.any(~finite):
            finite_idx = np.flatnonzero(finite)
            if len(finite_idx) < 2:
                raise LowQualitySignalError(f"RGB channel {i} has too few finite samples to interpolate.")
            channel[~finite] = np.interp(np.flatnonzero(~finite), finite_idx, channel[finite])
        out[:, i] = zscore_safe(smoothness_priors_detrend(channel, smoothness_lambda_for_fps(cfg, fps)))
    return out
