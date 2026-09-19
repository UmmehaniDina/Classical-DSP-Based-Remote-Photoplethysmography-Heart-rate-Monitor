import csv
import json

import pytest

from rppg.evaluation import evaluate_manifest, evaluate_reference_series, evaluate_reviewed_annotations, evaluate_session


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_session_reports_coverage_failures_and_bpm_error(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    _write_csv(
        csv_dir / "frame_events.csv",
        ["face_detected", "sample_accepted", "rejection_reason"],
        [
            {"face_detected": "True", "sample_accepted": "True", "rejection_reason": ""},
            {"face_detected": "False", "sample_accepted": "False", "rejection_reason": "no_landmarks"},
        ],
    )
    _write_csv(csv_dir / "bpm_estimates.csv", ["smoothed_bpm"], [{"smoothed_bpm": "70"}, {"smoothed_bpm": "74"}])

    report = evaluate_session(tmp_path, ground_truth_bpm=72.0)

    assert report["face_detection_availability"] == pytest.approx(0.5)
    assert report["sample_acceptance_rate"] == pytest.approx(0.5)
    assert report["mean_absolute_bpm_error"] == pytest.approx(2.0)
    assert report["rejection_reasons"] == {"no_landmarks": 1}


def test_evaluate_manifest_preserves_conditions_and_annotation(tmp_path):
    csv_dir = tmp_path / "session" / "csv"
    csv_dir.mkdir(parents=True)
    _write_csv(csv_dir / "frame_events.csv", ["face_detected", "sample_accepted", "rejection_reason"], [])
    _write_csv(csv_dir / "bpm_estimates.csv", ["smoothed_bpm"], [])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"id": "clip", "session_dir": str(csv_dir.parent), "video": "clip.mp4", "reviewed_annotation": "review.csv", "conditions": {"lighting": "dim"}}]}), encoding="utf-8")

    report = evaluate_manifest(manifest)

    assert report["clips"][0]["conditions"] == {"lighting": "dim"}
    assert report["clips"][0]["reviewed_annotation"] == "review.csv"


def test_evaluation_reports_face_loss_motion_and_reviewed_agreement(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    _write_csv(
        csv_dir / "frame_events.csv",
        ["frame_id", "face_detected", "sample_accepted", "rejection_reason", "skin_pixel_count"],
        [
            {"frame_id": "1", "face_detected": "True", "sample_accepted": "True", "rejection_reason": "", "skin_pixel_count": "80"},
            {"frame_id": "2", "face_detected": "False", "sample_accepted": "False", "rejection_reason": "no_landmarks", "skin_pixel_count": "0"},
            {"frame_id": "3", "face_detected": "True", "sample_accepted": "False", "rejection_reason": "landmark_motion_exceeds_limit", "skin_pixel_count": "70"},
        ],
    )
    _write_csv(csv_dir / "bpm_estimates.csv", ["smoothed_bpm"], [])
    annotations = tmp_path / "review.csv"
    _write_csv(
        annotations,
        ["frame_id", "expected_sample_accepted"],
        [{"frame_id": "1", "expected_sample_accepted": "true"}, {"frame_id": "3", "expected_sample_accepted": "true"}],
    )

    report = evaluate_session(tmp_path)
    agreement = evaluate_reviewed_annotations(tmp_path, annotations)

    assert report["face_loss_count"] == 1
    assert report["face_reacquisition_count"] == 1
    assert report["motion_rejection_count"] == 1
    assert report["roi_quality_fields_present"] is True
    assert agreement == {"reviewed_frame_count": 2, "acceptance_precision": 1.0, "acceptance_recall": 0.5}


def test_reference_series_interpolates_only_within_reference_times(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    _write_csv(
        csv_dir / "bpm_estimates.csv",
        ["time_sec", "smoothed_bpm"],
        [
            {"time_sec": "-1", "smoothed_bpm": "50"},
            {"time_sec": "0", "smoothed_bpm": "60"},
            {"time_sec": "1", "smoothed_bpm": "72"},
            {"time_sec": "2", "smoothed_bpm": "84"},
            {"time_sec": "3", "smoothed_bpm": "100"},
            {"time_sec": "bad", "smoothed_bpm": "nan"},
        ],
    )
    reference = tmp_path / "reference.csv"
    _write_csv(
        reference,
        ["time_sec", "bpm"],
        [{"time_sec": "0", "bpm": "60"}, {"time_sec": "2", "bpm": "80"}],
    )

    metrics = evaluate_reference_series(tmp_path, reference)

    assert metrics["paired_estimate_count"] == 3
    assert metrics["mean_absolute_bpm_error"] == pytest.approx(2.0)
    assert metrics["root_mean_squared_bpm_error"] == pytest.approx((20 / 3) ** 0.5)
    assert metrics["mean_bpm_bias"] == pytest.approx(2.0)
