import pytest

from rppg.calibration import CalibrationAccumulator


def test_calibration_profile_summarizes_quality_evidence():
    calibration = CalibrationAccumulator()
    calibration.add(True, True, 100, 0.50, 0.01, temporal_profile_ready=True)
    calibration.add(True, False, 80, 0.40, 0.02)
    calibration.add(False, False, 0, 0.0, float("nan"))

    profile = calibration.as_dict(10.0)

    assert profile["frame_count"] == 3
    assert profile["face_detection_availability"] == pytest.approx(2 / 3)
    assert profile["sample_acceptance_rate"] == pytest.approx(1 / 3)
    assert profile["skin_pixel_count"]["median"] == pytest.approx(80.0)
    assert profile["landmark_motion"]["p90"] == pytest.approx(0.019)
    assert profile["temporal_profile_ready_rate"] == pytest.approx(1 / 3)
