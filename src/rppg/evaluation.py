"""Manifest-driven offline evaluation of recorded rPPG sessions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(row: dict[str, str], field: str) -> float | None:
    """Return a finite numeric CSV field, or ``None`` when it is unusable."""
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _paired_metrics(estimated_bpm: np.ndarray, reference_bpm: np.ndarray) -> dict[str, float | int]:
    """Calculate non-clinical error summaries for already time-aligned BPM pairs."""
    if estimated_bpm.shape != reference_bpm.shape or estimated_bpm.size == 0:
        return {"paired_estimate_count": 0}

    error = estimated_bpm - reference_bpm
    result: dict[str, float | int] = {
        "paired_estimate_count": int(error.size),
        "mean_absolute_bpm_error": float(np.mean(np.abs(error))),
        "root_mean_squared_bpm_error": float(np.sqrt(np.mean(np.square(error)))),
        "mean_bpm_bias": float(np.mean(error)),
    }
    if error.size >= 2 and np.std(estimated_bpm) > 0.0 and np.std(reference_bpm) > 0.0:
        result["pearson_correlation"] = float(np.corrcoef(estimated_bpm, reference_bpm)[0, 1])
    return result


def evaluate_reference_series(session_dir: Path, reference_path: Path) -> dict[str, float | int]:
    """Compare estimates with a timestamped reference CSV on the same time base.

    The reference CSV must provide ``time_sec`` and ``bpm`` columns. Reference
    BPM is linearly interpolated only between its first and last timestamp;
    estimates outside that interval and non-finite values are excluded. This
    function does not synchronize separate clocks or validate a reference
    device, so callers must establish alignment before using its metrics.
    """
    estimates = _read_csv(Path(session_dir) / "csv" / "bpm_estimates.csv")
    reference = _read_csv(Path(reference_path))
    estimate_pairs = [
        (time, bpm)
        for row in estimates
        if (time := _finite_float(row, "time_sec")) is not None
        and (bpm := _finite_float(row, "smoothed_bpm")) is not None
    ]
    reference_pairs = [
        (time, bpm)
        for row in reference
        if (time := _finite_float(row, "time_sec")) is not None and (bpm := _finite_float(row, "bpm")) is not None
    ]
    if not estimate_pairs or len(reference_pairs) < 2:
        return {"paired_estimate_count": 0}

    reference_pairs.sort(key=lambda item: item[0])
    ref_times = np.asarray([item[0] for item in reference_pairs], dtype=np.float64)
    ref_bpm = np.asarray([item[1] for item in reference_pairs], dtype=np.float64)
    unique_times, inverse = np.unique(ref_times, return_inverse=True)
    if unique_times.size < 2:
        return {"paired_estimate_count": 0}
    if unique_times.size != ref_times.size:
        totals = np.bincount(inverse, weights=ref_bpm)
        counts = np.bincount(inverse)
        ref_times = unique_times
        ref_bpm = totals / counts

    estimate_times = np.asarray([item[0] for item in estimate_pairs], dtype=np.float64)
    estimate_bpm = np.asarray([item[1] for item in estimate_pairs], dtype=np.float64)
    in_range = (estimate_times >= ref_times[0]) & (estimate_times <= ref_times[-1])
    if not np.any(in_range):
        return {"paired_estimate_count": 0}
    aligned_reference = np.interp(estimate_times[in_range], ref_times, ref_bpm)
    return _paired_metrics(estimate_bpm[in_range], aligned_reference)


def evaluate_session(session_dir: Path, ground_truth_bpm: float | None = None) -> dict[str, Any]:
    """Summarize availability, failures, and optional BPM error for one replay session."""
    csv_dir = Path(session_dir) / "csv"
    frames = _read_csv(csv_dir / "frame_events.csv")
    estimates = _read_csv(csv_dir / "bpm_estimates.csv")
    frame_count = len(frames)
    accepted = sum(row.get("sample_accepted", "").lower() == "true" for row in frames)
    face_detected = sum(row.get("face_detected", "").lower() == "true" for row in frames)
    failures = Counter(row.get("rejection_reason", "") for row in frames if row.get("rejection_reason", ""))
    face_loss_count = sum(
        row.get("face_detected", "").lower() != "true"
        and (index == 0 or frames[index - 1].get("face_detected", "").lower() == "true")
        for index, row in enumerate(frames)
    )
    reacquisition_count = sum(
        row.get("face_detected", "").lower() == "true"
        and index > 0
        and frames[index - 1].get("face_detected", "").lower() != "true"
        for index, row in enumerate(frames)
    )
    motion_rejection_count = failures.get("landmark_motion_exceeds_limit", 0)
    bpm_values = np.asarray(
        [value for row in estimates if (value := _finite_float(row, "smoothed_bpm")) is not None],
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "session_dir": str(session_dir),
        "frame_count": frame_count,
        "face_detection_availability": face_detected / frame_count if frame_count else 0.0,
        "sample_acceptance_rate": accepted / frame_count if frame_count else 0.0,
        "bpm_estimate_count": len(bpm_values),
        "rejection_reasons": dict(sorted(failures.items())),
        "face_loss_count": face_loss_count,
        "face_reacquisition_count": reacquisition_count,
        "motion_rejection_count": motion_rejection_count,
        "roi_quality_fields_present": bool(frames and "skin_pixel_count" in frames[0]),
    }
    if ground_truth_bpm is not None and len(bpm_values) and np.isfinite(ground_truth_bpm):
        result.update(_paired_metrics(bpm_values, np.full_like(bpm_values, ground_truth_bpm)))
    return result


def evaluate_reviewed_annotations(session_dir: Path, annotation_path: Path) -> dict[str, float | int] | None:
    """Compare recorded sample acceptance with manually reviewed frame labels.

    The annotation CSV must contain ``frame_id`` and ``expected_sample_accepted``
    columns, where the latter is ``true`` or ``false``. It is intentionally a
    human review contract, not an inferred skin-tone label.
    """
    if not annotation_path.exists():
        return None
    events = {row.get("frame_id"): row for row in _read_csv(Path(session_dir) / "csv" / "frame_events.csv")}
    annotations = _read_csv(annotation_path)
    true_positive = false_positive = false_negative = 0
    reviewed = 0
    for annotation in annotations:
        expected = annotation.get("expected_sample_accepted", "").lower()
        event = events.get(annotation.get("frame_id"))
        if expected not in {"true", "false"} or event is None:
            continue
        reviewed += 1
        actual = event.get("sample_accepted", "").lower() == "true"
        wanted = expected == "true"
        true_positive += int(actual and wanted)
        false_positive += int(actual and not wanted)
        false_negative += int(not actual and wanted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {"reviewed_frame_count": reviewed, "acceptance_precision": precision, "acceptance_recall": recall}


def evaluate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Evaluate completed offline replay sessions listed in a JSON manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    clips = manifest.get("clips", [])
    results = []
    for clip in clips:
        session_dir = Path(clip["session_dir"])
        result = evaluate_session(session_dir, clip.get("ground_truth_bpm"))
        result["clip_id"] = clip["id"]
        result["conditions"] = clip.get("conditions", {})
        result["video"] = clip.get("video")
        result["reviewed_annotation"] = clip.get("reviewed_annotation")
        result["reference_bpm_csv"] = clip.get("reference_bpm_csv")
        if clip.get("reviewed_annotation"):
            result["annotation_agreement"] = evaluate_reviewed_annotations(
                session_dir,
                Path(clip["reviewed_annotation"]),
            )
        if clip.get("reference_bpm_csv"):
            result["reference_series_metrics"] = evaluate_reference_series(
                session_dir,
                Path(clip["reference_bpm_csv"]),
            )
        results.append(result)
    return {"manifest": str(manifest_path), "clip_count": len(results), "clips": results}


def main(argv: list[str] | None = None) -> int:
    """Write a JSON report for a manifest of completed offline replay sessions."""
    parser = argparse.ArgumentParser(description="Evaluate recorded rPPG replay sessions.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSON manifest describing replay sessions.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report path.")
    args = parser.parse_args(argv)
    report = evaluate_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
