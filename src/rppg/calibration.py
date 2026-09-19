"""Stationary-session quality calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CalibrationAccumulator:
    """Collect a descriptive baseline without changing live signal thresholds."""

    frame_count: int = 0
    face_detected_count: int = 0
    sample_accepted_count: int = 0
    skin_pixel_counts: list[int] = field(default_factory=list)
    mask_coverages: list[float] = field(default_factory=list)
    landmark_motions: list[float] = field(default_factory=list)
    temporal_profile_ready_count: int = 0

    def add(
        self,
        face_detected: bool,
        sample_accepted: bool,
        skin_pixel_count: int,
        mask_coverage: float,
        landmark_motion: float,
        temporal_profile_ready: bool = False,
    ) -> None:
        """Record frame-quality evidence for the baseline profile."""
        self.frame_count += 1
        self.face_detected_count += int(face_detected)
        self.sample_accepted_count += int(sample_accepted)
        self.temporal_profile_ready_count += int(temporal_profile_ready)
        self.skin_pixel_counts.append(skin_pixel_count)
        if np.isfinite(mask_coverage):
            self.mask_coverages.append(mask_coverage)
        if np.isfinite(landmark_motion):
            self.landmark_motions.append(landmark_motion)

    def as_dict(self, duration_sec: float) -> dict[str, Any]:
        """Return a JSON-safe descriptive profile for later review, not auto-tuning."""
        def summary(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"median": None, "p10": None, "p90": None}
            array = np.asarray(values, dtype=np.float64)
            return {
                "median": float(np.median(array)),
                "p10": float(np.percentile(array, 10)),
                "p90": float(np.percentile(array, 90)),
            }

        return {
            "duration_sec": duration_sec,
            "frame_count": self.frame_count,
            "face_detection_availability": self.face_detected_count / self.frame_count if self.frame_count else 0.0,
            "sample_acceptance_rate": self.sample_accepted_count / self.frame_count if self.frame_count else 0.0,
            "skin_pixel_count": summary([float(value) for value in self.skin_pixel_counts]),
            "mask_coverage": summary(self.mask_coverages),
            "landmark_motion": summary(self.landmark_motions),
            "temporal_profile_ready_rate": (
                self.temporal_profile_ready_count / self.frame_count if self.frame_count else 0.0
            ),
            "interpretation": (
                "Descriptive quality evidence only. This profile does not validate accuracy, fairness, "
                "or medical suitability and does not change thresholds automatically."
            ),
        }
