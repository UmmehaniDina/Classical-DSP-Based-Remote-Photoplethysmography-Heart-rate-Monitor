"""Session output directory and metadata helpers."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AppConfig
from .results import AnalysisResult


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def categorize_rejection_reason(reason: str, status: str) -> str:
    """Return a stable analysis-outcome category while preserving raw text."""
    if status == "accepted":
        return "accepted"
    if status == "warming":
        return "warmup"
    text = str(reason).lower()
    if "timestamp gap" in text or "accepted-sample gap" in text or "frame timestamp gap" in text:
        return "timing_gap"
    if "coverage" in text:
        return "coverage"
    if "stale" in text:
        return "stale_data"
    if "illumination" in text:
        return "illumination"
    if "motion" in text:
        return "motion"
    if "low-quality" in text or "quality" in text or "snr" in text:
        return "signal_quality"
    if status == "error":
        return "analysis_error"
    return "other_rejection"


def config_snapshot(cfg: AppConfig) -> dict[str, Any]:
    """Return a JSON-serializable configuration snapshot."""
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)


def create_session_directory(output_root: Path) -> Path:
    """Create logs, csv, figures, and metadata folders for a run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(output_root) / stamp
    for name in ("logs", "csv", "figures", "metadata"):
        (session_dir / name).mkdir(parents=True, exist_ok=True)
    return session_dir


def write_session_metadata(session_dir: Path, cfg: AppConfig) -> Path:
    """Write session metadata, configuration, and platform notes."""
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config_snapshot(cfg),
    }
    target = session_dir / "metadata" / "session.json"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return target


def write_calibration_profile(session_dir: Path, profile: dict[str, Any]) -> Path:
    """Write a descriptive stationary-session quality profile."""
    target = session_dir / "metadata" / "calibration.json"
    target.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return target


class CsvWriters:
    """CSV sink for frame events, accepted RGB samples, and BPM estimates."""

    def __init__(self, session_dir: Path) -> None:
        csv_dir = session_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        self._rgb_file = (csv_dir / "rgb_samples.csv").open("w", newline="", encoding="utf-8")
        self._bpm_file = (csv_dir / "bpm_estimates.csv").open("w", newline="", encoding="utf-8")
        self._frames_file = (csv_dir / "frame_events.csv").open("w", newline="", encoding="utf-8")
        self._analysis_file = (csv_dir / "analysis_events.csv").open("w", newline="", encoding="utf-8")
        self._rgb = csv.writer(self._rgb_file)
        self._bpm = csv.writer(self._bpm_file)
        self._frames = csv.writer(self._frames_file)
        self._analysis = csv.writer(self._analysis_file)
        self._rgb.writerow(["time_sec", "r", "g", "b"])
        self._bpm.writerow(
            [
                "time_sec",
                "smoothed_bpm",
                "fft_bpm",
                "peak_bpm",
                "quality",
                "candidate",
                "effective_fps",
                "band_low_hz",
                "band_high_hz",
                "illumination_instability",
                "candidate_metadata_json",
                "quality_features_json",
            ]
        )
        self._frames.writerow(
            [
                "frame_id",
                "time_sec",
                "timestamp_source",
                "face_detected",
                "sample_accepted",
                "status",
                "rejection_reason",
                "face_status",
                "roi_count",
                "valid_roi_count",
                "skin_pixel_count",
                "mask_coverage",
                "landmark_motion",
                "skin_mask_strategy",
                "temporal_profile_ready",
            ]
        )
        self._analysis.writerow(
            [
                "time_sec",
                "analysis_sequence",
                "status",
                "rejection_category",
                "rejection_reason",
                "sample_count",
                "effective_fps",
                "detection_availability",
                "sample_coverage",
                "candidate",
                "quality",
                "fused_bpm",
            ]
        )

    def write_frame_event(
        self,
        frame_id: int,
        time_sec: float,
        timestamp_source: str,
        face_detected: bool,
        sample_accepted: bool,
        status: str,
        rejection_reason: str = "",
        face_status: str = "",
        roi_count: int | None = None,
        valid_roi_count: int | None = None,
        skin_pixel_count: int | None = None,
        mask_coverage: float | None = None,
        landmark_motion: float | None = None,
        skin_mask_strategy: str = "fixed",
        temporal_profile_ready: bool = False,
    ) -> None:
        """Append one acquisition event, including rejected frames."""
        self._frames.writerow(
            [
                frame_id,
                time_sec,
                timestamp_source,
                face_detected,
                sample_accepted,
                status,
                rejection_reason,
                face_status,
                roi_count,
                valid_roi_count,
                skin_pixel_count,
                mask_coverage,
                landmark_motion,
                skin_mask_strategy,
                temporal_profile_ready,
            ]
        )
        self._frames_file.flush()

    def write_rgb(self, time_sec: float, rgb: Any) -> None:
        self._rgb.writerow([time_sec, float(rgb[0]), float(rgb[1]), float(rgb[2])])
        self._rgb_file.flush()

    def write_bpm(self, time_sec: float, result: AnalysisResult, smoothed_bpm: float) -> None:
        """Append one accepted BPM estimate to the session CSV."""
        self._bpm.writerow(
            [
                time_sec,
                smoothed_bpm,
                result.bpm_fft,
                result.bpm_peaks,
                result.quality,
                result.name,
                result.fps,
                result.band[0],
                result.band[1],
                result.illumination_instability,
                json.dumps(result.candidate_metadata, sort_keys=True, default=_json_default),
                json.dumps(result.quality_features, sort_keys=True, default=_json_default),
            ]
        )
        self._bpm_file.flush()

    def write_analysis_event(
        self,
        time_sec: float,
        analysis_sequence: int,
        status: str,
        *,
        rejection_reason: str = "",
        sample_count: int | None = None,
        effective_fps: float | None = None,
        detection_availability: float | None = None,
        sample_coverage: float | None = None,
        result: AnalysisResult | None = None,
    ) -> None:
        """Append one structured analysis-window outcome for post-session audit."""
        self._analysis.writerow(
            [
                time_sec,
                analysis_sequence,
                status,
                categorize_rejection_reason(rejection_reason, status),
                rejection_reason,
                sample_count,
                effective_fps,
                detection_availability,
                sample_coverage,
                "" if result is None else result.name,
                "" if result is None else result.quality,
                "" if result is None else result.fused_bpm,
            ]
        )
        self._analysis_file.flush()

    def close(self) -> None:
        self._rgb_file.close()
        self._bpm_file.close()
        self._frames_file.close()
        self._analysis_file.close()
