"""Skin masking and RGB extraction from ROI boxes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import RoiConfig
from .roi import RoiRegion

logger = logging.getLogger("rppg_monitor")


@dataclass(frozen=True)
class RoiValidity:
    """Validity evidence for one ROI in one frame."""

    box: tuple[int, int, int, int] | None
    pixel_count: int
    skin_pixel_count: int
    mask_coverage: float
    accepted: bool
    rejection_reason: str = ""
    fixed_skin_pixel_count: int = 0
    adaptive_skin_pixel_count: int = 0
    temporal_skin_pixel_count: int = 0


@dataclass(frozen=True)
class RgbExtraction:
    """RGB sample plus per-ROI evidence used to accept or reject it."""

    rgb: NDArray[np.float64]
    roi_validity: tuple[RoiValidity, ...]
    skin_mask_strategy: str = "fixed"
    temporal_profile_ready: bool = False

    @property
    def accepted(self) -> bool:
        """Whether the RGB sample is finite and has passed skin-pixel checks."""
        return bool(np.all(np.isfinite(self.rgb)))


@dataclass
class AdaptiveChromaProfile:
    """Conservative temporal Cr/Cb bounds learned only from fixed-mask seeds.

    This is a per-session heuristic, not a skin-tone classifier or learned
    segmentation model. It activates only after several valid fixed-mask
    observations and falls back to the current-frame adaptive mask otherwise.
    """

    bounds: dict[str, np.ndarray] = field(default_factory=dict)
    update_counts: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        """Discard session-specific chroma state after prolonged face loss."""
        self.bounds.clear()
        self.update_counts.clear()

    def mask(
        self,
        roi_bgr: NDArray[np.uint8],
        fixed_mask: NDArray[np.bool_],
        fallback_mask: NDArray[np.bool_],
        cfg: RoiConfig,
        key: str,
    ) -> tuple[NDArray[np.bool_], bool]:
        """Return a temporally stabilized mask and whether its profile is ready."""
        if fixed_mask.size == 0 or int(np.count_nonzero(fixed_mask)) < cfg.min_valid_skin_pixels_per_roi:
            return fallback_mask, False
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        low = cfg.adaptive_chroma_percentile
        high = 100.0 - low
        cr_low, cr_high = np.percentile(ycrcb[:, :, 1][fixed_mask], [low, high])
        cb_low, cb_high = np.percentile(ycrcb[:, :, 2][fixed_mask], [low, high])
        observed = np.asarray([cr_low, cr_high, cb_low, cb_high], dtype=np.float64)
        previous = self.bounds.get(key)
        alpha = cfg.temporal_adaptive_alpha
        self.bounds[key] = observed if previous is None else alpha * observed + (1.0 - alpha) * previous
        self.update_counts[key] = self.update_counts.get(key, 0) + 1
        ready = self.update_counts[key] >= cfg.temporal_adaptive_min_updates
        if not ready:
            return fallback_mask, False
        cr_low, cr_high, cb_low, cb_high = self.bounds[key]
        padding = cfg.adaptive_chroma_padding
        mask = (
            (ycrcb[:, :, 0] > cfg.skin_y_min)
            & (ycrcb[:, :, 1] >= max(0, cr_low - padding))
            & (ycrcb[:, :, 1] <= min(255, cr_high + padding))
            & (ycrcb[:, :, 2] >= max(0, cb_low - padding))
            & (ycrcb[:, :, 2] <= min(255, cb_high + padding))
        )
        return mask, True


def skin_mask_ycrcb(roi_bgr: NDArray[np.uint8], cfg: RoiConfig) -> NDArray[np.bool_]:
    """Return a conservative YCrCb skin mask for a BGR ROI."""
    if (
        roi_bgr is None
        or getattr(roi_bgr, "size", 0) == 0
        or roi_bgr.ndim != 3
        or roi_bgr.shape[2] != 3
    ):
        return np.zeros((0, 0), dtype=bool)

    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    return (
        (y > cfg.skin_y_min)
        & (cr >= cfg.skin_cr_min)
        & (cr <= cfg.skin_cr_max)
        & (cb >= cfg.skin_cb_min)
        & (cb <= cfg.skin_cb_max)
    )


def adaptive_skin_mask_ycrcb(
    roi_bgr: NDArray[np.uint8],
    fixed_mask: NDArray[np.bool_],
    cfg: RoiConfig,
) -> NDArray[np.bool_]:
    """Expand a fixed-mask seed using local robust Cr/Cb bounds.

    This optional heuristic is a comparison strategy, not a skin-tone model.
    It falls back to the fixed mask when the fixed seed is insufficient.
    """
    if fixed_mask.size == 0 or int(np.count_nonzero(fixed_mask)) < cfg.min_valid_skin_pixels_per_roi:
        return fixed_mask
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    low = cfg.adaptive_chroma_percentile
    high = 100.0 - low
    cr_low, cr_high = np.percentile(cr[fixed_mask], [low, high])
    cb_low, cb_high = np.percentile(cb[fixed_mask], [low, high])
    padding = cfg.adaptive_chroma_padding
    return (
        (ycrcb[:, :, 0] > cfg.skin_y_min)
        & (cr >= max(0, cr_low - padding))
        & (cr <= min(255, cr_high + padding))
        & (cb >= max(0, cb_low - padding))
        & (cb <= min(255, cb_high + padding))
    )


def robust_channel_mean(values: ArrayLike, cfg: RoiConfig) -> float:
    """Mean after 10th to 90th percentile clipping."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan

    low, high = np.percentile(
        values,
        [cfg.robust_mean_low_percentile, cfg.robust_mean_high_percentile],
    )
    if not np.isfinite(low) or not np.isfinite(high):
        return np.nan
    mean_value = float(np.mean(np.clip(values, low, high)))
    return mean_value if np.isfinite(mean_value) else np.nan


def extract_rgb_from_roi(
    frame_bgr: NDArray[np.uint8],
    roi_boxes: Iterable[Sequence[float]],
    cfg: RoiConfig,
) -> NDArray[np.float64]:
    """Extract mean RGB from valid skin pixels inside accepted ROI boxes."""
    return extract_rgb_with_metadata(frame_bgr, roi_boxes, cfg).rgb


def extract_rgb_with_metadata(
    frame_bgr: NDArray[np.uint8],
    roi_boxes: Iterable[Sequence[float] | RoiRegion],
    cfg: RoiConfig,
    adaptive_profile: AdaptiveChromaProfile | None = None,
) -> RgbExtraction:
    """Extract RGB and per-ROI validity metadata without unsafe pixel fallback."""
    nan_rgb = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    validity: list[RoiValidity] = []
    if (
        frame_bgr is None
        or getattr(frame_bgr, "size", 0) == 0
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
    ):
        return RgbExtraction(nan_rgb, tuple(validity))
    if not roi_boxes:
        return RgbExtraction(nan_rgb, tuple(validity))

    fh, fw = frame_bgr.shape[:2]
    b_vals = []
    g_vals = []
    r_vals = []
    total_valid_pixels = 0

    profile_ready = False
    for index, item in enumerate(roi_boxes):
        region = item if isinstance(item, RoiRegion) else None
        box = region.box if region is not None else item
        if box is None:
            validity.append(RoiValidity(None, 0, 0, 0.0, False, "missing_box"))
            continue
        try:
            x, y, w, h = [int(round(float(v))) for v in box]
        except (TypeError, ValueError):
            logger.debug("Skipping invalid ROI box: %r", box)
            validity.append(RoiValidity(None, 0, 0, 0.0, False, "invalid_box"))
            continue
        if w <= 0 or h <= 0 or w * h < cfg.min_valid_roi_area_pixels:
            logger.debug("Skipping ROI box with invalid dimensions.")
            validity.append(RoiValidity((x, y, w, h), 0, 0, 0.0, False, "invalid_area"))
            continue

        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(fw, x + w)
        y_end = min(fh, y + h)
        if x_start >= x_end or y_start >= y_end:
            logger.debug("Skipping ROI box outside frame after clamping.")
            validity.append(RoiValidity((x, y, w, h), 0, 0, 0.0, False, "outside_frame"))
            continue

        roi = frame_bgr[y_start:y_end, x_start:x_end]
        if roi.size == 0 or roi.ndim != 3 or roi.shape[2] != 3:
            logger.debug("Skipping empty or malformed ROI slice.")
            validity.append(RoiValidity((x_start, y_start, x_end - x_start, y_end - y_start), 0, 0, 0.0, False, "empty_roi"))
            continue

        fixed_mask = skin_mask_ycrcb(roi, cfg)
        if fixed_mask.size == 0 or fixed_mask.shape != roi.shape[:2]:
            logger.debug("Skipping ROI because skin mask is invalid.")
            validity.append(RoiValidity((x_start, y_start, x_end - x_start, y_end - y_start), roi.shape[0] * roi.shape[1], 0, 0.0, False, "invalid_skin_mask"))
            continue

        adaptive_mask = adaptive_skin_mask_ycrcb(roi, fixed_mask, cfg)
        temporal_mask = adaptive_mask
        temporal_ready = False
        if cfg.skin_mask_strategy == "temporal_adaptive" and adaptive_profile is not None:
            temporal_mask, temporal_ready = adaptive_profile.mask(
                roi,
                fixed_mask,
                adaptive_mask,
                cfg,
                region.name if region is not None else f"box_{index}",
            )
        if cfg.skin_mask_strategy == "adaptive_chroma":
            mask = adaptive_mask
        elif cfg.skin_mask_strategy == "temporal_adaptive":
            mask = temporal_mask
        else:
            mask = fixed_mask

        if region is not None and cfg.use_landmark_polygon_masks:
            polygon_mask = np.zeros(mask.shape, dtype=np.uint8)
            local_polygon = region.polygon - np.array([x_start, y_start], dtype=np.int32)
            cv2.fillConvexPoly(polygon_mask, local_polygon, 1)
            mask &= polygon_mask.astype(bool)

        fixed_pixels = int(np.count_nonzero(fixed_mask))
        adaptive_pixels = int(np.count_nonzero(adaptive_mask))
        temporal_pixels = int(np.count_nonzero(temporal_mask))
        profile_ready = profile_ready or temporal_ready
        valid_pixels = int(np.count_nonzero(mask))
        pixel_count = int(mask.size)
        coverage = valid_pixels / pixel_count if pixel_count else 0.0
        if valid_pixels < cfg.min_valid_skin_pixels_per_roi:
            logger.debug("Skipping ROI: only %d valid skin pixels.", valid_pixels)
            validity.append(RoiValidity((x_start, y_start, x_end - x_start, y_end - y_start), pixel_count, valid_pixels, coverage, False, "insufficient_skin_pixels", fixed_pixels, adaptive_pixels, temporal_pixels))
            continue

        b_vals.append(roi[:, :, 0][mask])
        g_vals.append(roi[:, :, 1][mask])
        r_vals.append(roi[:, :, 2][mask])
        total_valid_pixels += valid_pixels
        validity.append(RoiValidity((x_start, y_start, x_end - x_start, y_end - y_start), pixel_count, valid_pixels, coverage, True, "", fixed_pixels, adaptive_pixels, temporal_pixels))

    if total_valid_pixels < cfg.min_valid_skin_pixels_total:
        logger.debug("Rejecting frame sample: total valid skin pixels = %d", total_valid_pixels)
        return RgbExtraction(nan_rgb, tuple(validity))
    if not b_vals or not g_vals or not r_vals:
        return RgbExtraction(nan_rgb, tuple(validity))

    rgb = np.array(
        [
            robust_channel_mean(np.concatenate(r_vals), cfg),
            robust_channel_mean(np.concatenate(g_vals), cfg),
            robust_channel_mean(np.concatenate(b_vals), cfg),
        ],
        dtype=np.float64,
    )
    return RgbExtraction(rgb if np.all(np.isfinite(rgb)) else nan_rgb, tuple(validity), cfg.skin_mask_strategy, profile_ready)
