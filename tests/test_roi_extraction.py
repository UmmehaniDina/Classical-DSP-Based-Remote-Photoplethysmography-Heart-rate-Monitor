import numpy as np
import cv2
import pytest

from rppg.config import RoiConfig
from rppg.extraction import AdaptiveChromaProfile, extract_rgb_from_roi, extract_rgb_with_metadata
from rppg.roi import RoiBoxSmoother, RoiRegion, clamp_box_to_frame, landmark_motion
from rppg.session import CsvWriters


def test_clamp_box_to_frame_rejects_tiny_roi():
    assert clamp_box_to_frame([0, 0, 2, 2], (100, 100), min_area_pixels=50) is None


def test_clamp_box_to_frame_clamps_valid_roi():
    assert clamp_box_to_frame([-2, -3, 20, 20], (100, 100), min_area_pixels=50) == [0, 0, 18, 17]


def test_extract_rgb_rejects_without_skin_pixels():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb = extract_rgb_from_roi(frame, [[0, 0, 20, 20]], RoiConfig())
    assert np.all(np.isnan(rgb))


def test_extraction_reports_roi_validity_metadata():
    ycrcb = np.full((20, 20, 3), (100, 150, 110), dtype=np.uint8)
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    result = extract_rgb_with_metadata(frame, [[0, 0, 20, 20]], RoiConfig())

    assert result.accepted
    assert result.roi_validity[0].accepted
    assert result.roi_validity[0].skin_pixel_count >= 20
    assert result.roi_validity[0].mask_coverage > 0.0


def test_adaptive_strategy_reports_fixed_and_adaptive_mask_counts():
    ycrcb = np.full((20, 20, 3), (100, 150, 110), dtype=np.uint8)
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    result = extract_rgb_with_metadata(frame, [[0, 0, 20, 20]], RoiConfig(skin_mask_strategy="adaptive_chroma"))

    validity = result.roi_validity[0]
    assert result.skin_mask_strategy == "adaptive_chroma"
    assert validity.fixed_skin_pixel_count > 0
    assert validity.adaptive_skin_pixel_count > 0


def test_temporal_adaptive_strategy_activates_only_after_valid_seed_updates():
    ycrcb = np.full((20, 20, 3), (100, 150, 110), dtype=np.uint8)
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    cfg = RoiConfig(skin_mask_strategy="temporal_adaptive", temporal_adaptive_min_updates=2)
    profile = AdaptiveChromaProfile()

    first = extract_rgb_with_metadata(frame, [[0, 0, 20, 20]], cfg, profile)
    second = extract_rgb_with_metadata(frame, [[0, 0, 20, 20]], cfg, profile)

    assert first.temporal_profile_ready is False
    assert second.temporal_profile_ready is True
    assert second.roi_validity[0].temporal_skin_pixel_count > 0


def test_roi_box_smoother_reduces_coordinate_jump():
    cfg = RoiConfig(coordinate_smoothing_factor=0.5)
    smoother = RoiBoxSmoother(cfg.coordinate_smoothing_factor)
    polygon = np.array([[0, 0], [2, 0], [1, 2]], dtype=np.int32)
    first = smoother.smooth([RoiRegion("forehead", [10, 10, 20, 20], polygon)], (100, 100), cfg)
    second = smoother.smooth([RoiRegion("forehead", [30, 10, 20, 20], polygon)], (100, 100), cfg)

    assert first[0].box[0] == 10
    assert second[0].box[0] == 20


def test_landmark_motion_reports_median_normalized_displacement():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    motion = landmark_motion([Point(0.1, 0.1), Point(0.2, 0.2)], [Point(0.11, 0.1), Point(0.21, 0.2)])
    assert motion == pytest.approx(0.01)


def test_frame_events_record_rejected_and_accepted_frames(tmp_path):
    writers = CsvWriters(tmp_path)
    writers.write_frame_event(1, 0.0, "monotonic", False, False, "rejected", "no_landmarks")
    writers.write_frame_event(2, 0.1, "capture", True, True, "accepted")
    writers.close()

    rows = (tmp_path / "csv" / "frame_events.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("frame_id,time_sec,timestamp_source")
    assert "rejected,no_landmarks," in rows[1]
    assert ",accepted," in rows[2]


def test_bpm_csv_records_candidate_metadata(tmp_path):
    from rppg.results import AnalysisResult

    writers = CsvWriters(tmp_path)
    result = AnalysisResult(
        name="CHROM",
        quality=3.0,
        bpm_fft=72.0,
        bpm_peaks=72.0,
        fused_bpm=72.0,
        filtered=np.array([0.0, 1.0]),
        peaks=np.array([1]),
        freqs=np.array([1.2]),
        power=np.array([1.0]),
        uniform_times=np.array([0.0, 0.1]),
        fps=10.0,
        band=(0.9, 2.0),
        candidate_metadata={"alpha": 0.8, "formula": "test"},
        quality_features={"spectral_score": 3.0},
    )
    writers.write_bpm(1.0, result, 72.0)
    writers.close()

    header, row = (tmp_path / "csv" / "bpm_estimates.csv").read_text(encoding="utf-8").splitlines()
    assert "candidate_metadata_json" in header
    assert "quality_features_json" in header
    assert "effective_fps" in header
    assert "band_low_hz" in header
    assert "alpha" in row and "0.8" in row
