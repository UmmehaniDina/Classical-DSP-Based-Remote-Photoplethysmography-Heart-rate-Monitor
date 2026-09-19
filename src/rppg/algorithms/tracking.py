"""Optional BPM tracking filters."""

from __future__ import annotations

import math

import numpy as np

from rppg.config import AnalysisConfig


def time_aware_ema(previous_bpm: float, measured_bpm: float, dt_sec: float | None, cfg: AnalysisConfig) -> float:
    """Smooth BPM using elapsed time rather than a fixed update-count weight."""
    if not np.isfinite(previous_bpm):
        return float(measured_bpm)
    if not np.isfinite(measured_bpm):
        return float(previous_bpm)
    dt = cfg.analysis_update_interval_sec if dt_sec is None else max(float(dt_sec), 0.0)
    alpha = 1.0 - math.exp(-dt / cfg.bpm_smoothing_time_constant_sec)
    return float(alpha * measured_bpm + (1.0 - alpha) * previous_bpm)


class BpmKalmanTracker:
    """Constant-velocity Kalman tracker for accepted BPM measurements."""

    def __init__(self, cfg: AnalysisConfig) -> None:
        self._cfg = cfg
        self._x = np.array([cfg.kalman_initial_bpm, 0.0], dtype=np.float64)
        self._p = np.eye(2, dtype=np.float64) * cfg.kalman_initial_variance
        self._initialized = False

    def reset(self) -> None:
        """Forget tracker state after stale data or face-loss resets."""
        self._x[:] = [self._cfg.kalman_initial_bpm, 0.0]
        self._p = np.eye(2, dtype=np.float64) * self._cfg.kalman_initial_variance
        self._initialized = False

    def update(self, measured_bpm: float, dt: float | None = None) -> float:
        """Update the tracker and return the filtered BPM."""
        if not np.isfinite(measured_bpm):
            return float("nan")
        if not self._initialized:
            self._x[:] = [float(measured_bpm), 0.0]
            self._initialized = True
            return float(measured_bpm)

        dt_value = float(dt) if dt is not None and np.isfinite(dt) and dt > 0.0 else self._cfg.analysis_update_interval_sec
        transition = np.array([[1.0, dt_value], [0.0, 1.0]], dtype=np.float64)
        process = np.array(
            [
                [self._cfg.kalman_process_bpm_variance * dt_value, 0.0],
                [0.0, self._cfg.kalman_process_rate_variance * dt_value],
            ],
            dtype=np.float64,
        )

        self._x = transition @ self._x
        self._p = transition @ self._p @ transition.T + process

        observation = np.array([1.0, 0.0], dtype=np.float64)
        innovation = float(measured_bpm - (observation @ self._x))
        innovation_var = float(observation @ self._p @ observation.T + self._cfg.kalman_measurement_variance)
        if innovation_var <= 0.0 or not np.isfinite(innovation_var):
            return float(self._x[0])

        gain = (self._p @ observation.T) / innovation_var
        self._x = self._x + gain * innovation
        self._p = (np.eye(2, dtype=np.float64) - np.outer(gain, observation)) @ self._p
        return float(self._x[0])
