from dataclasses import replace

import numpy as np
import pytest

from rppg.algorithms.analysis import _uniform_rgb
from rppg.config import AnalysisConfig
from rppg.exceptions import LowQualitySignalError
from rppg.timing import CaptureClock, validate_frame_window


def test_capture_clock_uses_progressing_capture_timestamp_then_monotonic_fallback():
    clock = CaptureClock()
    first, first_source = clock.next_timestamp(20.0)
    second, second_source = clock.next_timestamp(20.0)

    assert first_source == "capture"
    assert second_source == "monotonic"
    assert second > first


def test_capture_clock_rejects_a_backward_capture_timestamp():
    clock = CaptureClock()
    first, _ = clock.next_timestamp(1000.0)
    second, source = clock.next_timestamp(20.0)

    assert source == "monotonic"
    assert second > first


def test_frame_window_metrics_accepts_complete_regular_window():
    cfg = AnalysisConfig()
    frame_times = np.arange(0.0, 2.0, 0.1)
    metrics = validate_frame_window(frame_times, [True] * len(frame_times), frame_times, 1.9, cfg)

    assert metrics.effective_fps == pytest.approx(10.0)
    assert metrics.detection_availability == pytest.approx(1.0)
    assert metrics.sample_coverage == pytest.approx(1.0)


def test_frame_window_rejects_large_accepted_sample_gap():
    cfg = replace(AnalysisConfig(), min_sample_coverage_fraction=0.1)
    frame_times = np.arange(0.0, 2.0, 0.1)
    sample_times = np.array([0.0, 0.1, 0.2, 1.5, 1.6, 1.7, 1.8, 1.9])

    with pytest.raises(LowQualitySignalError, match="Accepted-sample gap"):
        validate_frame_window(frame_times, [True] * len(frame_times), sample_times, 1.9, cfg)


def test_frame_window_rejects_stale_samples():
    cfg = replace(AnalysisConfig(), max_sample_age_sec=0.5, min_sample_coverage_fraction=0.1)
    frame_times = np.arange(0.0, 2.0, 0.1)
    sample_times = np.arange(0.0, 1.0, 0.1)

    with pytest.raises(LowQualitySignalError, match="stale"):
        validate_frame_window(frame_times, [True] * len(frame_times), sample_times, 1.9, cfg)


def test_uniform_rgb_aggregates_duplicate_timestamps():
    cfg = replace(AnalysisConfig(), min_interpolation_samples=2, min_uniform_samples=2, min_effective_fps=1.0)
    times, rgb, _ = _uniform_rgb(
        [0.0, 0.0, 0.1, 0.2],
        [[10.0, 20.0, 30.0], [30.0, 40.0, 50.0], [20.0, 30.0, 40.0], [25.0, 35.0, 45.0]],
        cfg,
    )

    assert np.all(np.diff(times) > 0.0)
    assert rgb[0, 0] == pytest.approx(20.0)


def test_uniform_rgb_rejects_large_interpolation_gap():
    cfg = replace(AnalysisConfig(), min_interpolation_samples=2, min_uniform_samples=2, min_effective_fps=1.0)

    with pytest.raises(LowQualitySignalError, match="interpolation limit"):
        _uniform_rgb(
            [0.0, 0.1, 0.2, 0.3, 1.5],
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
                [13.0, 23.0, 33.0],
                [14.0, 24.0, 34.0],
            ],
            cfg,
        )
