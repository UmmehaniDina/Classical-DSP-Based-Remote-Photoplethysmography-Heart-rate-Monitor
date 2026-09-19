"""Landmark-guided ROI geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import RoiConfig

FOREHEAD_LANDMARK_IDS = [10, 338, 297, 332, 284, 109, 67, 103, 54]
LEFT_CHEEK_LANDMARK_IDS = [117, 118, 100, 126, 205, 50]
RIGHT_CHEEK_LANDMARK_IDS = [346, 347, 329, 355, 425, 280]

RoiBox = list[int]


@dataclass(frozen=True)
class RoiRegion:
    """A named landmark-constrained ROI with its bounding box and polygon."""

    name: str
    box: RoiBox
    polygon: np.ndarray


class RoiBoxSmoother:
    """Exponentially smooth ROI boxes while retaining frame-boundary safety."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._boxes: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        """Discard prior tracking state after face loss or reacquisition."""
        self._boxes.clear()

    def smooth(self, regions: Sequence[RoiRegion], frame_size: tuple[int, int], cfg: RoiConfig) -> list[RoiRegion]:
        """Return regions whose boxes are smoothed against the prior frame."""
        smoothed: list[RoiRegion] = []
        for region in regions:
            current = np.asarray(region.box, dtype=np.float64)
            previous = self._boxes.get(region.name)
            box = current if previous is None else self._alpha * current + (1.0 - self._alpha) * previous
            clamped = clamp_box_to_frame(box, frame_size, cfg.min_valid_roi_area_pixels)
            if clamped is None:
                continue
            self._boxes[region.name] = np.asarray(clamped, dtype=np.float64)
            smoothed.append(RoiRegion(region.name, clamped, region.polygon))
        return smoothed


def landmark_motion(previous: Sequence[Any] | None, current: Sequence[Any] | None) -> float:
    """Return median normalized XY landmark displacement between two frames."""
    if previous is None or current is None or len(previous) != len(current) or not current:
        return float("nan")
    try:
        before = np.array([(point.x, point.y) for point in previous], dtype=np.float64)
        after = np.array([(point.x, point.y) for point in current], dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return float("nan")
    distance = np.linalg.norm(after - before, axis=1)
    return float(np.median(distance)) if np.all(np.isfinite(distance)) else float("nan")


def clamp_box_to_frame(
    box: Sequence[float] | None,
    frame_size: tuple[int, int] | None,
    min_area_pixels: int,
) -> RoiBox | None:
    """Clamp [x, y, w, h] to frame boundaries and reject invalid boxes."""
    if box is None or frame_size is None:
        return None
    try:
        fh, fw = frame_size
        x, y, w, h = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if fh <= 0 or fw <= 0 or w <= 0 or h <= 0:
        return None
    if not np.all(np.isfinite([x, y, w, h])):
        return None

    x_start = max(0, int(round(x)))
    y_start = max(0, int(round(y)))
    x_end = min(fw - 1, int(round(x + w - 1)))
    y_end = min(fh - 1, int(round(y + h - 1)))
    if x_start > x_end or y_start > y_end:
        return None

    width = x_end - x_start + 1
    height = y_end - y_start + 1
    if width * height < min_area_pixels:
        return None
    return [x_start, y_start, width, height]


def get_box_from_landmarks(
    landmarks: Sequence[Any],
    landmark_ids: Sequence[int],
    frame_width: int,
    frame_height: int,
    padding: int,
) -> RoiBox | None:
    """Build an ROI box from selected normalized MediaPipe landmarks."""
    try:
        x_pixels = [int(landmarks[landmark_id].x * frame_width) for landmark_id in landmark_ids]
        y_pixels = [int(landmarks[landmark_id].y * frame_height) for landmark_id in landmark_ids]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return [
        min(x_pixels) - padding,
        min(y_pixels) - padding,
        max(x_pixels) - min(x_pixels) + (padding * 2),
        max(y_pixels) - min(y_pixels) + (padding * 2),
    ]


def roi_boxes_from_landmarks(
    landmarks: Sequence[Any],
    frame_width: int,
    frame_height: int,
    cfg: RoiConfig,
) -> list[RoiBox]:
    """Return clamped ROI boxes for the configured ROI mode."""
    landmark_groups: list[Sequence[int]] = []
    if cfg.mode in ("forehead", "multi"):
        landmark_groups.append(FOREHEAD_LANDMARK_IDS)
    if cfg.mode in ("upperCheeks", "multi"):
        landmark_groups.extend([LEFT_CHEEK_LANDMARK_IDS, RIGHT_CHEEK_LANDMARK_IDS])

    boxes = [
        get_box_from_landmarks(
            landmarks,
            landmark_ids,
            frame_width,
            frame_height,
            cfg.landmark_padding_px,
        )
        for landmark_ids in landmark_groups
    ]
    return [
        box
        for box in (
            clamp_box_to_frame(box, (frame_height, frame_width), cfg.min_valid_roi_area_pixels)
            for box in boxes
        )
        if box is not None
    ]


def roi_regions_from_landmarks(
    landmarks: Sequence[Any],
    frame_width: int,
    frame_height: int,
    cfg: RoiConfig,
) -> list[RoiRegion]:
    """Return named boxes with landmark polygons for constrained skin sampling."""
    groups: list[tuple[str, Sequence[int]]] = []
    if cfg.mode in ("forehead", "multi"):
        groups.append(("forehead", FOREHEAD_LANDMARK_IDS))
    if cfg.mode in ("upperCheeks", "multi"):
        groups.extend((("left_cheek", LEFT_CHEEK_LANDMARK_IDS), ("right_cheek", RIGHT_CHEEK_LANDMARK_IDS)))

    regions: list[RoiRegion] = []
    for name, landmark_ids in groups:
        box = get_box_from_landmarks(landmarks, landmark_ids, frame_width, frame_height, cfg.landmark_padding_px)
        clamped = clamp_box_to_frame(box, (frame_height, frame_width), cfg.min_valid_roi_area_pixels)
        if clamped is None:
            continue
        try:
            points = np.array(
                [[landmarks[index].x * frame_width, landmarks[index].y * frame_height] for index in landmark_ids],
                dtype=np.float32,
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        if len(points) < 3 or not np.all(np.isfinite(points)):
            continue
        regions.append(RoiRegion(name, clamped, np.round(points).astype(np.int32)))
    return regions
