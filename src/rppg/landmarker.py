"""MediaPipe Tasks Face Landmarker integration."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from .config import AppConfig
from .exceptions import AcquisitionError, ConfigurationError

logger = logging.getLogger("rppg_monitor")


def ensure_face_landmarker_model(cfg: AppConfig) -> Path:
    """Ensure the Face Landmarker task bundle exists locally."""
    model_path = cfg.face_landmarker_model_path
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(".task.download")
    logger.info("Downloading MediaPipe Face Landmarker model to %s", model_path)

    try:
        urllib.request.urlretrieve(cfg.face_landmarker_model_url, temp_path)
        temp_path.replace(model_path)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                logger.debug("Could not remove temporary model file.", exc_info=True)
        raise AcquisitionError(
            "MediaPipe Tasks Face Landmarker needs a local model bundle. "
            f"Automatic download failed. Download {cfg.face_landmarker_model_url} "
            f"and save it as {model_path}."
        ) from exc

    return model_path


def create_face_landmarker(cfg: AppConfig) -> Any:
    """Create a MediaPipe VIDEO-mode FaceLandmarker."""
    if not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
        raise ConfigurationError(
            "This MediaPipe install does not include the Tasks vision API. "
            "Upgrade MediaPipe with: python -m pip install --upgrade mediapipe"
        )
    if not hasattr(mp.tasks, "BaseOptions"):
        raise ConfigurationError(
            "This MediaPipe install is missing mp.tasks.BaseOptions. "
            "Upgrade MediaPipe with: python -m pip install --upgrade mediapipe"
        )

    required = ("FaceLandmarker", "FaceLandmarkerOptions", "RunningMode")
    missing = [name for name in required if not hasattr(mp.tasks.vision, name)]
    if missing:
        raise ConfigurationError(
            "This MediaPipe install is missing required Face Landmarker classes: "
            f"{', '.join(missing)}."
        )

    model_path = ensure_face_landmarker_model(cfg)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=cfg.roi.min_face_detection_confidence,
        min_face_presence_confidence=cfg.roi.min_face_presence_confidence,
        min_tracking_confidence=cfg.roi.min_tracking_confidence,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def detect_face_landmarks(
    face_landmarker: Any,
    frame_rgb: NDArray[np.uint8],
    timestamp_ms: int,
) -> Any | None:
    """Return the first detected face landmark list, or None."""
    try:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception:
        logger.debug("Face landmark detection failed for frame.", exc_info=True)
        return None

    if results is None or not results.face_landmarks:
        return None
    return results.face_landmarks[0]
