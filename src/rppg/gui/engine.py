"""Background processing engine for the rPPG monitor GUI."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rppg.algorithms.analysis import analyze_window
from rppg.algorithms.tracking import BpmKalmanTracker, time_aware_ema
from rppg.config import AppConfig, parse_camera_source
from rppg.exceptions import AcquisitionError, AnalysisError, ConfigurationError, LowQualitySignalError, WarmupError
from rppg.extraction import AdaptiveChromaProfile, extract_rgb_with_metadata
from rppg.landmarker import create_face_landmarker, detect_face_landmarks
from rppg.results import AnalysisResult
from rppg.roi import RoiBoxSmoother, landmark_motion, roi_regions_from_landmarks
from rppg.session import CsvWriters, create_session_directory, write_session_metadata
from rppg.timing import CaptureClock, validate_frame_window
from rppg.ubfc import evaluate_ubfc_session, is_ubfc_dataset_folder, resolve_ubfc_dataset

logger = logging.getLogger("rppg_gui")


@dataclass
class GuiTelemetry:
    """Telemetry data payload dispatched to the GUI thread."""

    status_text: str
    face_detected: bool
    smoothed_bpm: float = float("nan")
    quality: float = float("nan")
    bpm_fft: float = float("nan")
    bpm_peaks: float = float("nan")
    fused_bpm: float = float("nan")
    filtered_signal: np.ndarray | None = None
    peaks: np.ndarray | None = None
    freqs: np.ndarray | None = None
    power: np.ndarray | None = None
    dom_freq: float | None = None
    effective_fps: float = 0.0
    total_frames: int = 0
    accepted_samples: int = 0
    ground_truth_bpm: float | None = None
    bpm_history_times: list[float] | None = None
    bpm_history_values: list[float] | None = None
    gt_history_values: list[float] | None = None
    has_ground_truth: bool = False
    active_method: str = ""
    skin_pixels: int | None = None
    landmark_motion: float | None = None
    session_dir: Path | None = None


class RppgEngine:
    """Threaded execution coordinator for the classical rPPG monitor."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2)
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._is_running = False
        self._session_dir: Path | None = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def get_frame(self) -> np.ndarray | None:
        """Poll the latest video frame without blocking."""
        frame = None
        while not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        return frame

    def get_events(self) -> list[dict[str, Any]]:
        """Poll all queued analysis events without blocking."""
        events = []
        while not self._event_queue.empty():
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def start(self, cfg: AppConfig) -> None:
        """Start the processing pipeline in a background thread."""
        if self._is_running:
            return
        self._stop_event.clear()
        # Drain any residual frames or events
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break

        self._is_running = True
        self._thread = threading.Thread(target=self._run_loop, args=(cfg,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker thread to terminate and wait."""
        if not self._is_running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._is_running = False

    def _send_event(self, event_type: str, **kwargs: Any) -> None:
        try:
            self._event_queue.put_nowait({"type": event_type, **kwargs})
        except queue.Full:
            pass

    def _send_frame(self, frame: np.ndarray) -> None:
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _run_loop(self, cfg: AppConfig) -> None:
        """Main worker execution loop matching rppg.app.runner."""
        os.environ.setdefault("MPLCONFIGDIR", str(cfg.project_root / ".mplconfig"))
        session_dir = create_session_directory(cfg.output_root)
        self._session_dir = session_dir
        write_session_metadata(session_dir, cfg)
        csv_writers = CsvWriters(session_dir)

        face_landmarker = None
        cap = None

        self._send_event("status", text="Initializing camera & detector...", color=None, session_dir=session_dir)

        try:
            face_landmarker = create_face_landmarker(cfg)
            from rppg.app.runner import open_capture, _is_offline_video_source

            cap = open_capture(cfg)
            offline_video = _is_offline_video_source(cfg)

            sample_times = deque()
            rgb_samples = deque()
            sample_mask_coverages = deque()
            sample_motion_qualities = deque()
            frame_times = deque()
            frame_face_detected = deque()
            bpm_history_times = deque()
            bpm_history_values = deque()
            gt_history_values = deque()
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
            roi_smoother = RoiBoxSmoother(cfg.roi.coordinate_smoothing_factor)
            adaptive_profile = (
                AdaptiveChromaProfile() if cfg.roi.skin_mask_strategy == "temporal_adaptive" else None
            )
            previous_landmarks = None
            face_lost_since: float | None = None
            bpm_tracker = BpmKalmanTracker(cfg.analysis) if cfg.analysis.use_bpm_kalman_tracker else None
            consecutive_analysis_rejections = 0

            # Ground truth reading for UBFC if present
            gt_ref_times: np.ndarray | None = None
            gt_ref_bpm: np.ndarray | None = None
            if is_ubfc_dataset_folder(cfg.camera.source):
                try:
                    from rppg.ubfc import load_ubfc_reference
                    gt_ref_times, gt_ref_bpm = load_ubfc_reference(cfg.camera.source)
                    logger.info("Loaded UBFC ground truth reference (%d samples)", len(gt_ref_times))
                except Exception as exc:
                    logger.warning("Failed to load UBFC reference: %s", exc)
                    gt_ref_times, gt_ref_bpm = None, None

            # Frame interval pacing for offline video files so playback runs at normal speed (not sped up)
            video_fps = cap.get(cv2.CAP_PROP_FPS) if offline_video else 0.0
            if offline_video:
                if not np.isfinite(video_fps) or video_fps <= 1.0 or video_fps > 120.0:
                    video_fps = 30.0
                frame_interval = 1.0 / video_fps
                start_playback_wall_time = time.monotonic()
                playback_frame_count = 0

            self._send_event("status", text="Acquiring face & pulse...", color=None, session_dir=session_dir)

            while not self._stop_event.is_set():
                if offline_video:
                    target_time = start_playback_wall_time + (playback_frame_count * frame_interval)
                    sleep_dur = target_time - time.monotonic()
                    if sleep_dur > 0.001:
                        if self._stop_event.wait(timeout=sleep_dur):
                            break
                    playback_frame_count += 1

                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None or frame_bgr.size == 0:
                    if offline_video:
                        self._send_event("status", text="Replay completed", color=None)
                        break
                    self._send_event("error", error="Camera frame read failed.")
                    break

                current_time, timestamp_source = capture_clock.next_timestamp(cap.get(cv2.CAP_PROP_POS_MSEC))
                frame_id += 1
                total_frames += 1
                first_frame_time = current_time if first_frame_time is None else first_frame_time
                last_frame_time = current_time
                h, w = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(int(round(current_time * 1000.0)), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                landmarks = detect_face_landmarks(face_landmarker, frame_rgb, timestamp_ms)
                display = frame_bgr.copy()
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

                    # Draw clean stylish ROI polygons on display
                    for region in regions:
                        roi_x, roi_y, roi_width, roi_height = region.box
                        # Cyan / Emerald rounded box
                        cv2.rectangle(
                            display,
                            (roi_x, roi_y),
                            (roi_x + roi_width, roi_y + roi_height),
                            (16, 185, 129),  # Emerald BGR
                            2,
                        )
                        if cfg.roi.debug_overlay:
                            cv2.polylines(display, [region.polygon], True, (244, 63, 94), 1)

                else:
                    rejection_reason = "no_landmarks"
                    if face_lost_since is None:
                        face_lost_since = current_time
                    previous_landmarks = None

                frame_times.append(current_time)
                frame_face_detected.append(face_detected)

                # Send display frame to GUI queue
                self._send_frame(display)

                # Trim history
                while sample_times and sample_times[0] < current_time - cfg.analysis.analysis_window_sec:
                    sample_times.popleft()
                    rgb_samples.popleft()
                    sample_mask_coverages.popleft()
                    sample_motion_qualities.popleft()
                while frame_times and frame_times[0] < current_time - cfg.analysis.analysis_window_sec:
                    frame_times.popleft()
                    frame_face_detected.popleft()

                data_is_stale = (
                    bool(sample_times) and current_time - sample_times[-1] > cfg.analysis.max_sample_age_sec
                )
                if data_is_stale:
                    smoothed_bpm = np.nan
                    last_valid_bpm = np.nan
                    last_analysis_quality = np.nan
                    last_bpm_update_time = None
                    if bpm_tracker is not None:
                        bpm_tracker.reset()

                # Status string
                if current_time < cfg.camera.warmup_sec:
                    status = f"Warming up camera ({cfg.camera.warmup_sec - current_time:.1f}s)..."
                elif data_is_stale:
                    status = "Face repositioning / waiting for samples..."
                elif not np.isfinite(smoothed_bpm):
                    status = "Buffering pulse window (calibrating)..."
                elif consecutive_analysis_rejections:
                    status = f"{smoothed_bpm:.1f} BPM (rechecking signal)"
                elif np.isfinite(last_analysis_quality):
                    status = f"{smoothed_bpm:.1f} BPM | SQI {last_analysis_quality:.2f}"
                else:
                    status = f"{smoothed_bpm:.1f} BPM"

                # DSP Window Analysis Step
                if current_time - last_analysis_time >= cfg.analysis.analysis_update_interval_sec:
                    last_analysis_time = current_time
                    if (
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

                            # Ground truth comparison if UBFC
                            gt_bpm = None
                            if gt_ref_times is not None and gt_ref_bpm is not None and len(gt_ref_times) > 0:
                                if current_time <= gt_ref_times[0]:
                                    gt_bpm = float(gt_ref_bpm[0])
                                elif current_time >= gt_ref_times[-1]:
                                    gt_bpm = float(gt_ref_bpm[-1])
                                else:
                                    gt_bpm = float(np.interp(current_time, gt_ref_times, gt_ref_bpm))
                            gt_history_values.append(gt_bpm if gt_bpm is not None else np.nan)
                            csv_writers.write_bpm(current_time, result, smoothed_bpm)

                            while bpm_history_times and bpm_history_times[0] < current_time - cfg.analysis.bpm_history_sec:
                                bpm_history_times.popleft()
                                bpm_history_values.popleft()
                                gt_history_values.popleft()

                            # Dispatch rich telemetry to GUI
                            telemetry = GuiTelemetry(
                                status_text=status,
                                face_detected=face_detected,
                                smoothed_bpm=smoothed_bpm,
                                quality=result.quality,
                                bpm_fft=result.bpm_fft,
                                bpm_peaks=result.bpm_peaks,
                                fused_bpm=result.fused_bpm,
                                filtered_signal=result.filtered,
                                peaks=result.peaks,
                                freqs=result.freqs,
                                power=result.power,
                                dom_freq=result.bpm_fft / 60.0 if np.isfinite(result.bpm_fft) else None,
                                effective_fps=metrics.effective_fps,
                                total_frames=total_frames,
                                accepted_samples=total_accepted_samples,
                                ground_truth_bpm=gt_bpm,
                                bpm_history_times=list(bpm_history_times),
                                bpm_history_values=list(bpm_history_values),
                                gt_history_values=list(gt_history_values),
                                has_ground_truth=(gt_ref_times is not None and gt_ref_bpm is not None),
                                active_method=result.name,
                                skin_pixels=total_skin_pixels,
                                landmark_motion=float(motion) if np.isfinite(motion) else None,
                                session_dir=session_dir,
                            )
                            self._send_event("telemetry", data=telemetry)

                        except WarmupError as exc:
                            self._send_event(
                                "telemetry",
                                data=GuiTelemetry(
                                    status_text=f"Warming: {exc}",
                                    face_detected=face_detected,
                                    effective_fps=total_frames / max(current_time - first_frame_time, 1e-3)
                                    if first_frame_time
                                    else 0.0,
                                    total_frames=total_frames,
                                    accepted_samples=total_accepted_samples,
                                    session_dir=session_dir,
                                ),
                            )
                        except LowQualitySignalError as exc:
                            consecutive_analysis_rejections += 1
                            self._send_event(
                                "telemetry",
                                data=GuiTelemetry(
                                    status_text=f"Low Quality: {exc}",
                                    face_detected=face_detected,
                                    effective_fps=total_frames / max(current_time - first_frame_time, 1e-3)
                                    if first_frame_time
                                    else 0.0,
                                    total_frames=total_frames,
                                    accepted_samples=total_accepted_samples,
                                    session_dir=session_dir,
                                ),
                            )
                        except Exception as exc:
                            logger.exception("Analysis error")
                    else:
                        # Warmup or buffer filling
                        self._send_event(
                            "telemetry",
                            data=GuiTelemetry(
                                status_text=status,
                                face_detected=face_detected,
                                effective_fps=total_frames / max(current_time - first_frame_time, 1e-3)
                                if first_frame_time
                                else 0.0,
                                total_frames=total_frames,
                                accepted_samples=total_accepted_samples,
                                session_dir=session_dir,
                            ),
                        )

            # End of session handling
            if is_ubfc_dataset_folder(cfg.camera.source):
                try:
                    report = evaluate_ubfc_session(session_dir, cfg.camera.source)
                    self._send_event("ubfc_summary", report=report)
                except Exception as exc:
                    logger.exception("UBFC evaluation failed: %s", exc)

            self._send_event("finished", session_dir=session_dir)

        except Exception as exc:
            logger.exception("Fatal error in rPPG engine")
            self._send_event("error", error=str(exc))
        finally:
            csv_writers.close()
            if face_landmarker is not None:
                try:
                    face_landmarker.close()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._is_running = False
