import csv
import json

import pytest

from rppg.ubfc import evaluate_ubfc_session, load_ubfc_reference, resolve_ubfc_dataset


def _make_dataset(tmp_path):
    dataset = tmp_path / "subject1"
    dataset.mkdir()
    (dataset / "vid.avi").write_bytes(b"")
    (dataset / "ground_truth.txt").write_text(
        "0.1 0.2 0.3 0.4\n60 62 64 66\n0 1000 2000 3000\n", encoding="utf-8"
    )
    return dataset


def test_load_ubfc_reference_reads_standard_three_row_ground_truth(tmp_path):
    dataset = _make_dataset(tmp_path)

    times, bpm = load_ubfc_reference(dataset)

    assert resolve_ubfc_dataset(dataset)[0].name == "vid.avi"
    assert times.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert bpm.tolist() == pytest.approx([60.0, 62.0, 64.0, 66.0])


def test_ubfc_evaluation_writes_per_estimate_errors_and_mae(tmp_path):
    dataset = _make_dataset(tmp_path)
    session = tmp_path / "session"
    csv_dir = session / "csv"
    metadata_dir = session / "metadata"
    csv_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    with (csv_dir / "bpm_estimates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_sec", "smoothed_bpm"])
        writer.writeheader()
        writer.writerows([{"time_sec": "1", "smoothed_bpm": "63"}, {"time_sec": "2", "smoothed_bpm": "61"}])

    report = evaluate_ubfc_session(session, dataset)

    assert report["paired_estimate_count"] == 2
    assert report["mean_absolute_bpm_error"] == pytest.approx(2.0)
    assert (csv_dir / "ubfc_bpm_comparison.csv").exists()
    assert (csv_dir / "ubfc_frame_error.csv").exists()
    assert json.loads((metadata_dir / "ubfc_evaluation.json").read_text(encoding="utf-8"))["paired_estimate_count"] == 2
