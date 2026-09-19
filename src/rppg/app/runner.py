"""Main application orchestration."""

from __future__ import annotations

import logging
import os
import platform
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from rppg.algorithms.analysis import analyze_window
from rppg.algorithms.tracking import BpmKalmanTracker, time_aware_ema
from rppg.calibration import CalibrationAccumulator
from rppg.config import AppConfig, parse_camera_source
from rppg.exceptions import AcquisitionError, AnalysisError, ConfigurationError, LowQualitySignalError, WarmupError
from rppg.extraction import AdaptiveChromaProfile, extract_rgb_with_metadata
from rppg.landmarker import create_face_landmarker, detect_face_landmarks
from rppg.roi import RoiBoxSmoother, landmark_motion, roi_regions_from_landmarks
from rppg.session import CsvWriters, create_session_directory, write_calibration_profile, write_session_metadata
from rppg.timing import CaptureClock, validate_frame_window
from rppg.ubfc import evaluate_ubfc_session, is_ubfc_dataset_folder, resolve_ubfc_dataset

logger = logging.getLogger("rppg_monitor")

BACKEND_FLAGS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "gstreamer": cv2.CAP_GSTREAMER,
}


def structured_fields(**fields: object) -> str:
    """Format log fields as stable key=value pairs."""
    parts = []
    for key, value in fields.items():
        if isinstance(value, float):
            value = f"{value:.3f}"
        text = str(value).replace(" ", "_")
        parts.append(f"{key}={text}")
    return " ".join(parts)


def configure_logging(log_level: str, session_dir: Path) -> None:
    """Configure console and session-file logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    log_file = session_dir / "logs" / "rppg_monitor.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | rppg: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)


def open_capture(cfg: AppConfig) -> cv2.VideoCapture:
    """Open OpenCV VideoCapture with a portable backend setting.

    Raises:
        AcquisitionError: If the configured source cannot be opened.
    """
    source = parse_camera_source(cfg.camera.source)
    if isinstance(source, str) and is_ubfc_dataset_folder(source):
        source = str(resolve_ubfc_dataset(source)[0])
    backend = cfg.camera.backend
    if backend != "auto":
        cap = cv2.VideoCapture(source, BACKEND_FLAGS[backend])
    else:
        if isinstance(source, int) and platform.system().lower() == "windows":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

    if not cap.isOpened() and backend != "auto":
        cap.release()
        logger.warning("Camera backend %s failed; retrying with OpenCV default.", backend)
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise AcquisitionError(f"Cannot open camera/video source: {cfg.camera.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.requested_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.requested_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.requested_fps)
    _apply_manual_camera_controls(cap, cfg)
    logger.info(
        "camera configured | %s",
        structured_fields(
            requested_width=cfg.camera.requested_width,
            requested_height=cfg.camera.requested_height,
            requested_fps=cfg.camera.requested_fps,
            actual_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            actual_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            actual_fps=cap.get(cv2.CAP_PROP_FPS),
        ),
    )
    return cap


def _apply_manual_camera_controls(cap: cv2.VideoCapture, cfg: AppConfig) -> None:
    """Request opt-in manual camera controls and log each backend result."""
    requested_controls = (
        ("exposure", cv2.CAP_PROP_EXPOSURE, cfg.camera.manual_exposure),
        ("gain", cv2.CAP_PROP_GAIN, cfg.camera.manual_gain),
        ("white_balance", cv2.CAP_PROP_WB_TEMPERATURE, cfg.camera.manual_white_balance),
        ("focus", cv2.CAP_PROP_FOCUS, cfg.camera.manual_focus),
    )
    if cfg.camera.manual_exposure is not None:
        auto_result = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        logger.info(
            "camera control | %s",
            structured_fields(control="auto_exposure", requested=0.25, accepted=auto_result),
        )
    for name, property_id, requested in requested_controls:
        if requested is None:
            continue
        accepted = cap.set(property_id, requested)
        logger.info(
            "camera control | %s",
            structured_fields(control=name, requested=requested, accepted=accepted, actual=cap.get(property_id)),
        )


def _is_offline_video_source(cfg: AppConfig) -> bool:
    """Return whether cfg points to a local video file that should end cleanly."""
    source = parse_camera_source(cfg.camera.source)
    return isinstance(source, str) and (Path(source).expanduser().is_file() or is_ubfc_dataset_folder(source))


def draw_status(display: np.ndarray, status: str) -> None:
    """Draw the current status string onto the live camera frame."""
    cv2.putText(
        display,
        f"HR: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )


def run(cfg: AppConfig) -> int:
    """Run the live rPPG application."""
    os.environ.setdefault("MPLCONFIGDIR", str(cfg.project_root / ".mplconfig"))
    session_dir = create_session_directory(cfg.output_root)
    configure_logging(cfg.log_level, session_dir)
    write_session_metadata(session_dir, cfg)
    csv_writers = CsvWriters(session_dir)

    face_landmarker = None
    cap = None
    dashboard = None

    logger.info("Session output: %s", session_dir)
    logger.info("Initializing classical MediaPipe Tasks rPPG monitor...")
    logger.info("Research prototype: estimated heart rate only; not a medical device.")

    try:
        face_landmarker = create_face_landmarker(cfg)
        cap = open_capture(cfg)

        if not cfg.headless:
            from rppg.visualization import Dashboard

            dashboard = Dashboard.create(cfg.analysis)
            logger.info("Press 'q' in the camera window to stop.")
        else:
            logger.info("Headless mode enabled; press Ctrl+C to stop.")

        sample_times = deque()
        rgb_samples = deque()
        sample_mask_coverages = deque()
        sample_motion_qualities = deque()
        frame_times = deque()
        frame_face_detected = deque()
        bpm_history_times = deque()
        bpm_history_values = deque()
        smoothed_bpm = np.nan
        last_valid_bpm = np.nan
        last_analysis_quality = np.nan
        last_analysis_time = -np.inf
        analysis_sequence = 0
        last_bpm_update_time: float | None = None
        capture_clock = CaptureClock()
        last_timestamp_ms = -1
        frame_id = 0
        total_frames = 0
        total_faces = 0
        total_accepted_samples = 0
        first_frame_time: float | None = None
        last_frame_time: float | None = None
        offline_video = _is_offline_video_source(cfg)
        roi_smoother = RoiBoxSmoother(cfg.roi.coordinate_smoothing_factor)
        adaptive_profile = AdaptiveChromaProfile() if cfg.roi.skin_mask_strategy == "temporal_adaptive" else None
        previous_landmarks = None
        face_lost_since: float | None = None
        calibration = CalibrationAccumulator() if cfg.calibration_duration_sec is not None else None
        calibration_started_at: float | None = None
        bpm_tracker = BpmKalmanTracker(cfg.analysis) if cfg.analysis.use_bpm_kalman_tracker else None
        consecutive_analysis_rejections = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None or frame_bgr.size == 0:
                if offline_video:
                    logger.info("Offline video replay completed.")
                    break
                raise AcquisitionError("Camera frame read failed.")

            current_time, timestamp_source = capture_clock.next_timestamp(cap.get(cv2.CAP_PROP_POS_MSEC))
            calibration_started_at = current_time if calibration_started_at is None else calibration_started_at
            frame_id += 1
            total_frames += 1
            first_frame_time = current_time if first_frame_time is None else first_frame_time
            last_frame_time = current_time
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = max(int(round(current_time * 1000.0)), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            landmarks = detect_face_landmarks(face_landmarker, frame_rgb, timestamp_ms)
            display = frame_bgr.copy() if not cfg.headless else None
            face_detected = landmarks is not None
            sample_accepted = False
            rejection_reason = ""
            face_status = "detected" if face_detected else "lost"
            roi_count = 0
            valid_roi_count = 0
            total_skin_pixels = 0
            mean_mask_coverage = 0.0
            motion = float("nan")

            if landmarks is not None:
                total_faces += 1
                if face_lost_since is not None and current_time - face_lost_since >= cfg.roi.face_loss_reset_sec:
                    sample_times.clear()
                    rgb_samples.clear()
                    sample_mask_coverages.clear()
                    sample_motion_qualities.clear()
                    roi_smoother.reset()
                    if adaptive_profile is not None:
                        adaptive_profile.reset()
                    previous_landmarks = None
                    logger.info(
                        "face reacquired | %s",
                        structured_fields(face_status="reacquired_reset", loss_duration=current_time - face_lost_since),
                    )
                face_lost_since = None
                motion = landmark_motion(previous_landmarks, landmarks)
                previous_landmarks = landmarks
                regions = roi_regions_from_landmarks(landmarks, w, h, cfg.roi)
                regions = roi_smoother.smooth(regions, (h, w), cfg.roi)
                roi_count = len(regions)
                extraction = extract_rgb_with_metadata(frame_bgr, regions, cfg.roi, adaptive_profile)
                rgb = extraction.rgb
                valid_roi_count = sum(item.accepted for item in extraction.roi_validity)
                total_skin_pixels = sum(item.skin_pixel_count for item in extraction.roi_validity)
                fixed_skin_pixels = sum(item.fixed_skin_pixel_count for item in extraction.roi_validity)
                adaptive_skin_pixels = sum(item.adaptive_skin_pixel_count for item in extraction.roi_validity)
                temporal_skin_pixels = sum(item.temporal_skin_pixel_count for item in extraction.roi_validity)
                mean_mask_coverage = (
                    float(np.mean([item.mask_coverage for item in extraction.roi_validity]))
                    if extraction.roi_validity
                    else 0.0
                )
                motion_rejected = np.isfinite(motion) and motion > cfg.roi.max_landmark_motion
                if motion_rejected:
                    face_status = "motion_rejected"
                if np.all(np.isfinite(rgb)) and not motion_rejected:
                    sample_times.append(current_time)
                    rgb_samples.append(rgb)
                    sample_mask_coverages.append(mean_mask_coverage)
                    sample_motion_qualities.append(
                        1.0
                        if not np.isfinite(motion)
                        else max(0.0, 1.0 - (motion / cfg.roi.max_landmark_motion))
                    )
                    csv_writers.write_rgb(current_time, rgb)
                    sample_accepted = True
                    total_accepted_samples += 1
                    logger.debug(
                        "frame accepted | %s",
                        structured_fields(
                            frame_status="accepted",
                            roi_count=len(regions),
                            valid_roi_count=valid_roi_count,
                            skin_pixels=total_skin_pixels,
                            skin_mask_strategy=extraction.skin_mask_strategy,
                            fixed_skin_pixels=fixed_skin_pixels,
                            adaptive_skin_pixels=adaptive_skin_pixels,
                            temporal_skin_pixels=temporal_skin_pixels,
                            temporal_profile_ready=extraction.temporal_profile_ready,
                            mask_coverage=mean_mask_coverage,
                            landmark_motion=motion,
                            sample_count=len(sample_times),
                        ),
                    )
                else:
                    rejection_reason = "landmark_motion_exceeds_limit" if motion_rejected else "rgb_sample_empty_or_non_finite"
                    logger.debug(
                        "frame rejected | %s",
                        structured_fields(
                            frame_status="rejected",
                            roi_count=len(regions),
                            valid_roi_count=valid_roi_count,
                            skin_pixels=total_skin_pixels,
                            skin_mask_strategy=extraction.skin_mask_strategy,
                            fixed_skin_pixels=fixed_skin_pixels,
                            adaptive_skin_pixels=adaptive_skin_pixels,
                            temporal_skin_pixels=temporal_skin_pixels,
                            temporal_profile_ready=extraction.temporal_profile_ready,
                            mask_coverage=mean_mask_coverage,
                            landmark_motion=motion,
                            rejection_reason=rejection_reason,
                        ),
                    )

                if display is not None:
                    for region in regions:
                        roi_x, roi_y, roi_width, roi_height = region.box
                        cv2.rectangle(
                            display,
                            (roi_x, roi_y),
                            (roi_x + roi_width, roi_y + roi_height),
                            (0, 255, 0),
                            2,
                        )
                        if cfg.roi.debug_overlay:
                            cv2.polylines(display, [region.polygon], True, (255, 255, 0), 1)
                    if cfg.roi.debug_overlay:
                        cv2.putText(
                            display,
                            f"skin={total_skin_pixels} motion={motion:.4f}",
                            (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )
            else:
                rejection_reason = "no_landmarks"
                if face_lost_since is None:
                    face_lost_since = current_time
                previous_landmarks = None
                logger.debug(
                    "frame rejected | %s",
                    structured_fields(
                        frame_status="rejected",
                        face_status="lost",
                        face_loss_duration=current_time - face_lost_since,
                        rejection_reason="no_landmarks",
                    ),
                )

            frame_times.append(current_time)
            frame_face_detected.append(face_detected)
            csv_writers.write_frame_event(
                frame_id,
                current_time,
                timestamp_source,
                face_detected,
                sample_accepted,
                "accepted" if sample_accepted else "rejected",
                rejection_reason,
                face_status,
                roi_count,
                valid_roi_count,
                total_skin_pixels,
                mean_mask_coverage,
                motion,
                extraction.skin_mask_strategy if landmarks is not None else cfg.roi.skin_mask_strategy,
                extraction.temporal_profile_ready if landmarks is not None else False,
            )
            if calibration is not None:
                calibration.add(
                    face_detected,
                    sample_accepted,
                    total_skin_pixels,
                    mean_mask_coverage,
                    motion,
                    extraction.temporal_profile_ready if landmarks is not None else False,
                )
                duration = current_time - calibration_started_at
                if duration >= cfg.calibration_duration_sec:
                    profile_path = write_calibration_profile(session_dir, calibration.as_dict(duration))
                    logger.info("Calibration completed | %s", structured_fields(profile_path=profile_path))
                    break

            while sample_times and sample_times[0] < current_time - cfg.analysis.analysis_window_sec:
                sample_times.popleft()
                rgb_samples.popleft()
                sample_mask_coverages.popleft()
                sample_motion_qualities.popleft()
            while frame_times and frame_times[0] < current_time - cfg.analysis.analysis_window_sec:
                frame_times.popleft()
                frame_face_detected.popleft()

            data_is_stale = bool(sample_times) and current_time - sample_times[-1] > cfg.analysis.max_sample_age_sec
            if data_is_stale:
                smoothed_bpm = np.nan
                last_valid_bpm = np.nan
                last_analysis_quality = np.nan
                last_bpm_update_time = None
                if bpm_tracker is not None:
                    bpm_tracker.reset()

            if current_time < cfg.camera.warmup_sec:
                status = "Camera warming up..."
            elif data_is_stale:
                status = "Waiting for recent data..."
            else:
                if not np.isfinite(smoothed_bpm):
                    status = "No valid estimate - warming up"
                elif consecutive_analysis_rejections:
                    status = f"{smoothed_bpm:.1f} BPM (rechecking quality)"
                elif np.isfinite(last_analysis_quality):
                    status = f"{smoothed_bpm:.1f} BPM | SQI {last_analysis_quality:.2f}"
                else:
                    status = f"{smoothed_bpm:.1f} BPM"
            if display is not None:
                draw_status(display, status)
                cv2.imshow("Live Camera", display)

            if current_time - last_analysis_time >= cfg.analysis.analysis_update_interval_sec:
                last_analysis_time = current_time
                if (
                    cfg.calibration_duration_sec is None
                    and
                    current_time >= cfg.camera.warmup_sec
                    and len(sample_times) > 1
                    and sample_times[-1] - sample_times[0] >= cfg.analysis.min_warmup_sec
                ):
                    try:
                        current_analysis_sequence = analysis_sequence
                        analysis_sequence += 1
                        metrics = validate_frame_window(
                            list(frame_times),
                            list(frame_face_detected),
                            list(sample_times),
                            current_time,
                            cfg.analysis,
                        )
                        result = analyze_window(
                            list(sample_times),
                            list(rgb_samples),
                            last_valid_bpm,
                            cfg.analysis,
                            analysis_sequence=current_analysis_sequence,
                            quality_context={
                                "sample_coverage": metrics.sample_coverage,
                                "detection_availability": metrics.detection_availability,
                                "mask_coverage": float(np.mean(sample_mask_coverages)),
                                "motion_quality": float(np.mean(sample_motion_qualities)),
                            },
                        )
                        current_bpm = result.fused_bpm
                        if not np.isfinite(current_bpm):
                            raise LowQualitySignalError("Fused BPM is non-finite.")
                        dt_bpm = (
                            None
                            if last_bpm_update_time is None
                            else max(current_time - last_bpm_update_time, 1e-6)
                        )
                        if bpm_tracker is not None:
                            smoothed_bpm = bpm_tracker.update(current_bpm, dt_bpm)
                        elif not np.isfinite(smoothed_bpm):
                            smoothed_bpm = current_bpm
                        else:
                            smoothed_bpm = time_aware_ema(smoothed_bpm, current_bpm, dt_bpm, cfg.analysis)
                        last_bpm_update_time = current_time
                        consecutive_analysis_rejections = 0
                        last_analysis_quality = result.quality

                        last_valid_bpm = smoothed_bpm
                        bpm_history_times.append(current_time)
                        bpm_history_values.append(smoothed_bpm)
                        csv_writers.write_bpm(current_time, result, smoothed_bpm)
                        csv_writers.write_analysis_event(
                            current_time,
                            current_analysis_sequence,
                            "accepted",
                            sample_count=len(sample_times),
                            effective_fps=metrics.effective_fps,
                            detection_availability=metrics.detection_availability,
                            sample_coverage=metrics.sample_coverage,
                            result=result,
                        )

                        while bpm_history_times and bpm_history_times[0] < current_time - cfg.analysis.bpm_history_sec:
                            bpm_history_times.popleft()
                            bpm_history_values.popleft()

                        logger.info(
                            "analysis accepted | %s",
                            structured_fields(
                                frame_status="accepted",
                                fps=result.fps,
                                quality=result.quality,
                                candidate=result.name,
                                effective_fps=metrics.effective_fps,
                                detection_availability=metrics.detection_availability,
                                sample_coverage=metrics.sample_coverage,
                                illumination_instability=result.illumination_instability,
                                smoothed_bpm=smoothed_bpm,
                                bpm_fft=result.bpm_fft,
                                bpm_peaks=result.bpm_peaks,
                            ),
                        )

                        if dashboard is not None:
                            dashboard.update(
                                sample_times,
                                rgb_samples,
                                bpm_history_times,
                                bpm_history_values,
                                current_time,
                                result,
                                smoothed_bpm,
                            )
                    except WarmupError as exc:
                        csv_writers.write_analysis_event(
                            current_time,
                            current_analysis_sequence,
                            "warming",
                            rejection_reason=str(exc),
                            sample_count=len(sample_times),
                        )
                        logger.info(
                            "analysis warming | %s",
                            structured_fields(
                                frame_status="warming",
                                rejection_reason=str(exc),
                                sample_count=len(sample_times),
                            ),
                        )
                    except LowQualitySignalError as exc:
                        rejection = str(exc)
                        consecutive_analysis_rejections += 1
                        csv_writers.write_analysis_event(
                            current_time,
                            current_analysis_sequence,
                            "rejected",
                            rejection_reason=rejection,
                            sample_count=len(sample_times),
                        )
                        logger.info(
                            "analysis rejected | %s",
                            structured_fields(
                                frame_status="rejected",
                                rejection_reason=rejection,
                                sample_count=len(sample_times),
                            ),
                        )
                        if "timestamp gap" in rejection.lower():
                            sample_times.clear()
                            rgb_samples.clear()
                            sample_mask_coverages.clear()
                            sample_motion_qualities.clear()
                            frame_times.clear()
                            frame_face_detected.clear()
                            smoothed_bpm = np.nan
                            last_valid_bpm = np.nan
                            last_analysis_quality = np.nan
                            logger.info(
                                "acquisition window reset | %s",
                                structured_fields(rejection_reason="timestamp_gap", recovery="warmup_restarted"),
                            )
                        elif consecutive_analysis_rejections >= cfg.analysis.max_consecutive_low_quality_rejections:
                            smoothed_bpm = np.nan
                            last_valid_bpm = np.nan
                            last_analysis_quality = np.nan
                            last_bpm_update_time = None
                            if bpm_tracker is not None:
                                bpm_tracker.reset()
                            logger.info(
                                "BPM tracker reset | %s",
                                structured_fields(
                                    rejection_reason="repeated_low_quality",
                                    rejected_windows=consecutive_analysis_rejections,
                                ),
                            )
                            consecutive_analysis_rejections = 0
                    except AnalysisError as exc:
                        csv_writers.write_analysis_event(
                            current_time,
                            current_analysis_sequence,
                            "error",
                            rejection_reason=str(exc),
                            sample_count=len(sample_times),
                        )
                        logger.warning(
                            "analysis expected_error | %s",
                            structured_fields(frame_status="error", rejection_reason=str(exc)),
                        )
                    except Exception:
                        logger.exception("Unexpected error during analysis.")

            if not cfg.headless and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if cfg.benchmark and first_frame_time is not None and last_frame_time is not None:
            duration = max(last_frame_time - first_frame_time, np.finfo(float).eps)
            logger.info(
                "benchmark summary | %s",
                structured_fields(
                    frame_count=total_frames,
                    effective_fps=(total_frames - 1) / duration if total_frames > 1 else 0.0,
                    detection_availability=total_faces / total_frames if total_frames else 0.0,
                    sample_coverage=total_accepted_samples / total_frames if total_frames else 0.0,
                ),
            )

        if is_ubfc_dataset_folder(cfg.camera.source):
            report = evaluate_ubfc_session(session_dir, cfg.camera.source)
            logger.info(
                "UBFC evaluation | %s",
                structured_fields(
                    paired_estimate_count=report["paired_estimate_count"],
                    mean_absolute_bpm_error=report.get("mean_absolute_bpm_error", float("nan")),
                ),
            )
            from rppg.visualization import show_ubfc_final_results

            show_ubfc_final_results(report, show_window=not cfg.headless)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except (ConfigurationError, AcquisitionError) as exc:
        logger.error("Controlled startup/runtime failure | %s", structured_fields(rejection_reason=str(exc)))
        return 1
    except Exception:
        logger.exception("Unexpected fatal error in main loop.")
        return 1
    finally:
        csv_writers.close()
        if face_landmarker is not None:
            try:
                face_landmarker.close()
                logger.info("FaceLandmarker closed.")
            except Exception:
                logger.exception("Failed to close FaceLandmarker.")
        if cap is not None:
            try:
                cap.release()
                logger.info("Camera released.")
            except Exception:
                logger.exception("Failed to release camera.")
        if not cfg.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                logger.exception("Failed to destroy OpenCV windows.")
        if dashboard is not None:
            try:
                dashboard.close()
                logger.info("Matplotlib figure closed.")
            except Exception:
                logger.exception("Failed to close Matplotlib figure.")
    return 0
