import pytest

from rppg.algorithms.tracking import BpmKalmanTracker, time_aware_ema
from rppg.config import AnalysisConfig


def test_kalman_tracker_initializes_from_first_measurement():
    tracker = BpmKalmanTracker(AnalysisConfig())

    assert tracker.update(72.0, dt=1.0) == pytest.approx(72.0)


def test_kalman_tracker_smooths_measurement_jump():
    tracker = BpmKalmanTracker(AnalysisConfig())
    tracker.update(72.0, dt=1.0)

    filtered = tracker.update(100.0, dt=1.0)

    assert 72.0 < filtered < 100.0


def test_kalman_tracker_reset_forgets_prior_state():
    tracker = BpmKalmanTracker(AnalysisConfig())
    tracker.update(72.0, dt=1.0)
    tracker.update(100.0, dt=1.0)

    tracker.reset()

    assert tracker.update(80.0, dt=1.0) == pytest.approx(80.0)


def test_time_aware_ema_changes_more_for_longer_elapsed_time():
    cfg = AnalysisConfig(bpm_smoothing_time_constant_sec=5.0)

    short_gap = time_aware_ema(70.0, 100.0, 1.0, cfg)
    long_gap = time_aware_ema(70.0, 100.0, 5.0, cfg)

    assert 70.0 < short_gap < long_gap < 100.0
