"""Typed configuration and validation for the rPPG monitor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Literal, cast

from rppg.exceptions import ConfigurationError

RoiMode = Literal["forehead", "upperCheeks", "multi"]
SkinMaskStrategy = Literal["fixed", "adaptive_chroma", "temporal_adaptive"]
CameraBackend = Literal["auto", "dshow", "msmf", "v4l2", "avfoundation", "gstreamer"]


@dataclass(frozen=True)
class CameraConfig:
    """Camera or video acquisition settings."""

    source: str = "0"
    requested_width: int = 640
    requested_height: int = 480
    requested_fps: int = 30
    backend: CameraBackend = "auto"
    warmup_sec: float = 2.0
    manual_exposure: float | None = None
    manual_gain: float | None = None
    manual_white_balance: float | None = None
    manual_focus: float | None = None


@dataclass(frozen=True)
class RoiConfig:
    """Landmark ROI mode and skin-pixel acceptance thresholds."""

    mode: RoiMode = "multi"
    min_valid_roi_area_pixels: int = 50
    min_valid_skin_pixels_per_roi: int = 5
    min_valid_skin_pixels_total: int = 20
    landmark_padding_px: int = 5
    skin_y_min: int = 35
    skin_cr_min: int = 133
    skin_cr_max: int = 180
    skin_cb_min: int = 77
    skin_cb_max: int = 135
    skin_mask_strategy: SkinMaskStrategy = "fixed"
    adaptive_chroma_percentile: float = 10.0
    adaptive_chroma_padding: int = 8
    temporal_adaptive_alpha: float = 0.20
    temporal_adaptive_min_updates: int = 12
    robust_mean_low_percentile: float = 10.0
    robust_mean_high_percentile: float = 90.0
    coordinate_smoothing_factor: float = 0.45
    use_landmark_polygon_masks: bool = True
    max_landmark_motion: float = 0.025
    face_loss_reset_sec: float = 2.0
    debug_overlay: bool = False
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class AnalysisConfig:
    """Heart-rate estimation, filtering, and smoothing settings."""

    min_accepted_bpm: float = 55.0
    max_accepted_bpm: float = 200.0
    adaptive_half_bandwidth_hz: float = 0.55
    min_adaptive_bandwidth_hz: float = 0.90
    analysis_window_sec: float = 24.0
    min_warmup_sec: float = 16.0
    analysis_update_interval_sec: float = 1.0
    butterworth_order: int = 3
    filter_padtype: str = "odd"
    filter_transient_trim_sec: float = 0.0
    bpm_smoothing_factor: float = 0.18
    bpm_smoothing_time_constant_sec: float = 5.0
    bpm_history_sec: float = 120.0
    smoothness_priors_lambda: float = 80.0
    smoothness_priors_reference_fps: float = 30.0
    min_spectral_snr: float = 2.0
    max_bpm_jump_per_update: float = 18.0
    max_consecutive_low_quality_rejections: int = 3
    min_effective_fps: float = 8.0
    min_uniform_samples: int = 30
    min_interpolation_samples: int = 5
    candidate_projection_window_sec: float = 1.6
    welch_window_sec: float = 8.0
    welch_min_segment_samples: int = 64
    welch_target_resolution_hz: float = 0.10
    peak_prominence_std_fraction: float = 0.20
    min_peak_prominence: float = 0.05
    peak_detect_both_polarities: bool = True
    peak_match_tolerance_bpm: float = 18.0
    peak_fusion_fft_weight: float = 0.70
    peak_fusion_peak_weight: float = 0.30
    spectral_peak_exclusion_hz: float = 0.12
    spectral_refine_peak: bool = True
    spectral_edge_penalty_hz: float = 0.08
    spectral_edge_penalty: float = 0.65
    harmonic_power_fraction: float = 0.55
    subharmonic_tolerance_hz: float = 0.08
    harmonic_tolerance_hz: float = 0.08
    harmonic_quality_bonus: float = 1.10
    min_band_low_hz: float = 0.05
    nyquist_high_fraction: float = 0.95
    pca_min_component_std: float = 1e-6
    pca_sign_min_abs_correlation: float = 0.10
    pca_sign_min_samples: int = 8
    enable_ica_candidate: bool = False
    use_bpm_kalman_tracker: bool = False
    kalman_initial_bpm: float = 75.0
    kalman_initial_variance: float = 100.0
    kalman_process_bpm_variance: float = 4.0
    kalman_process_rate_variance: float = 1.0
    kalman_measurement_variance: float = 16.0
    global_band_probe_enabled: bool = True
    global_band_probe_interval_updates: int = 10
    global_band_probe_min_quality_ratio: float = 1.15
    max_timestamp_gap_factor: float = 8.0
    min_sample_coverage_fraction: float = 0.70
    max_sample_age_sec: float = 2.0
    max_interpolation_gap_factor: float = 8.0
    max_illumination_instability: float = 0.08
    extraction_method: str = "all"

    @property
    def global_search_band_hz(self) -> tuple[float, float]:
        """Global BPM search band converted to Hz."""
        return (self.min_accepted_bpm / 60.0, self.max_accepted_bpm / 60.0)


@dataclass(frozen=True)
class AppConfig:
    """Top-level application settings."""

    project_root: Path
    model_dir: Path
    face_landmarker_model_path: Path
    face_landmarker_model_url: str
    output_root: Path
    headless: bool
    benchmark: bool
    calibration_duration_sec: float | None
    log_level: str
    camera: CameraConfig
    roi: RoiConfig
    analysis: AnalysisConfig


def project_root_from_package() -> Path:
    """Return the repository root for the src-layout package."""
    return Path(__file__).resolve().parents[2]


def default_config() -> AppConfig:
    """Build defaults that match the original prototype behavior."""
    project_root = project_root_from_package()
    model_dir = project_root / "models"
    model_path = model_dir / "face_landmarker.task"
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    return AppConfig(
        project_root=project_root,
        model_dir=model_dir,
        face_landmarker_model_path=model_path,
        face_landmarker_model_url=model_url,
        output_root=project_root / "outputs",
        headless=False,
        benchmark=False,
        calibration_duration_sec=None,
        log_level="INFO",
        camera=CameraConfig(),
        roi=RoiConfig(),
        analysis=AnalysisConfig(),
    )


def with_overrides(
    cfg: AppConfig,
    *,
    source: str | None = None,
    roi_mode: str | None = None,
    skin_mask_strategy: str | None = None,
    headless: bool | None = None,
    benchmark: bool | None = None,
    calibration_duration_sec: float | None = None,
    debug_roi: bool | None = None,
    output_dir: str | Path | None = None,
    log_level: str | None = None,
    camera_backend: str | None = None,
    manual_exposure: float | None = None,
    manual_gain: float | None = None,
    manual_white_balance: float | None = None,
    manual_focus: float | None = None,
    analysis_window_sec: float | None = None,
    min_warmup_sec: float | None = None,
    candidate_projection_window_sec: float | None = None,
    global_band_probe_interval_updates: int | None = None,
    extraction_method: str | None = None,
) -> AppConfig:
    """Return a copy of cfg with CLI overrides applied."""
    camera = cfg.camera
    roi = cfg.roi
    analysis = cfg.analysis

    if source is not None:
        camera = replace(camera, source=str(source))
    if camera_backend is not None:
        camera = replace(camera, backend=cast(CameraBackend, str(camera_backend).lower()))
    if manual_exposure is not None:
        camera = replace(camera, manual_exposure=float(manual_exposure))
    if manual_gain is not None:
        camera = replace(camera, manual_gain=float(manual_gain))
    if manual_white_balance is not None:
        camera = replace(camera, manual_white_balance=float(manual_white_balance))
    if manual_focus is not None:
        camera = replace(camera, manual_focus=float(manual_focus))
    if roi_mode is not None:
        roi = replace(roi, mode=cast(RoiMode, str(roi_mode)))
    if skin_mask_strategy is not None:
        roi = replace(roi, skin_mask_strategy=cast(SkinMaskStrategy, str(skin_mask_strategy)))
    if debug_roi is not None:
        roi = replace(roi, debug_overlay=bool(debug_roi))
    if analysis_window_sec is not None:
        analysis = replace(analysis, analysis_window_sec=float(analysis_window_sec))
    if min_warmup_sec is not None:
        analysis = replace(analysis, min_warmup_sec=float(min_warmup_sec))
    if candidate_projection_window_sec is not None:
        analysis = replace(analysis, candidate_projection_window_sec=float(candidate_projection_window_sec))
    if global_band_probe_interval_updates is not None:
        analysis = replace(analysis, global_band_probe_interval_updates=int(global_band_probe_interval_updates))
    if extraction_method is not None:
        meth = str(extraction_method).strip().lower()
        enable_ica = (meth == "ica") or cfg.analysis.enable_ica_candidate
        analysis = replace(analysis, extraction_method=meth, enable_ica_candidate=enable_ica)

    resolved_benchmark = cfg.benchmark if benchmark is None else bool(benchmark)
    return replace(
        cfg,
        camera=camera,
        roi=roi,
        analysis=analysis,
        headless=(cfg.headless if headless is None else bool(headless)) or resolved_benchmark,
        benchmark=resolved_benchmark,
        calibration_duration_sec=calibration_duration_sec,
        output_root=cfg.output_root if output_dir is None else Path(output_dir),
        log_level=cfg.log_level if log_level is None else str(log_level).upper(),
    )


def parse_camera_source(source: str) -> int | str:
    """Convert a numeric source string to a camera index, otherwise return a path/URI."""
    value = str(source).strip()
    if value == "":
        raise ConfigurationError("Camera source must not be empty.")
    if value.isdigit():
        return int(value)
    return value


def validate_config(cfg: AppConfig) -> None:
    """Validate settings before opening hardware or processing frames."""
    allowed_roi_modes = {"forehead", "upperCheeks", "multi"}
    if cfg.roi.mode not in allowed_roi_modes:
        raise ConfigurationError(f"ROI mode must be one of {sorted(allowed_roi_modes)}.")

    allowed_backends = {"auto", "dshow", "msmf", "v4l2", "avfoundation", "gstreamer"}
    if cfg.camera.backend not in allowed_backends:
        raise ConfigurationError(f"Camera backend must be one of {sorted(allowed_backends)}.")

    source = parse_camera_source(cfg.camera.source)
    if isinstance(source, int) and source < 0:
        raise ConfigurationError("Camera source index must be >= 0.")
    if isinstance(source, str) and "://" not in source:
        source_path = Path(source).expanduser()
        if not source_path.exists():
            raise ConfigurationError(f"Camera/video source does not exist: {source}")
        if source_path.is_dir():
            from rppg.ubfc import resolve_ubfc_dataset

            resolve_ubfc_dataset(source_path)

    if cfg.camera.requested_width <= 0 or cfg.camera.requested_height <= 0:
        raise ConfigurationError("Requested camera dimensions must be positive.")
    if cfg.camera.requested_fps <= 0:
        raise ConfigurationError("Requested camera FPS must be positive.")
    if cfg.camera.warmup_sec < 0.0:
        raise ConfigurationError("Camera warmup length must be non-negative.")
    if cfg.calibration_duration_sec is not None and cfg.calibration_duration_sec <= 0.0:
        raise ConfigurationError("Calibration duration must be positive.")
    for name, value in (
        ("manual exposure", cfg.camera.manual_exposure),
        ("manual gain", cfg.camera.manual_gain),
        ("manual white balance", cfg.camera.manual_white_balance),
        ("manual focus", cfg.camera.manual_focus),
    ):
        if value is not None and not isfinite(value):
            raise ConfigurationError(f"Requested {name} must be finite.")

    roi = cfg.roi
    if roi.min_valid_roi_area_pixels <= 0:
        raise ConfigurationError("Minimum ROI area must be positive.")
    if roi.min_valid_skin_pixels_per_roi <= 0:
        raise ConfigurationError("Minimum skin pixels per ROI must be positive.")
    if roi.min_valid_skin_pixels_total <= 0:
        raise ConfigurationError("Minimum total skin pixels must be positive.")
    if roi.landmark_padding_px < 0:
        raise ConfigurationError("ROI landmark padding must be non-negative.")
    if not (0 <= roi.skin_y_min <= 255):
        raise ConfigurationError("Skin Y threshold must be in [0, 255].")
    if not (0 <= roi.skin_cr_min <= roi.skin_cr_max <= 255):
        raise ConfigurationError("Skin Cr thresholds must satisfy 0 <= min <= max <= 255.")
    if not (0 <= roi.skin_cb_min <= roi.skin_cb_max <= 255):
        raise ConfigurationError("Skin Cb thresholds must satisfy 0 <= min <= max <= 255.")
    if roi.skin_mask_strategy not in {"fixed", "adaptive_chroma", "temporal_adaptive"}:
        raise ConfigurationError("Skin mask strategy must be fixed, adaptive_chroma, or temporal_adaptive.")
    if not (0.0 <= roi.adaptive_chroma_percentile < 50.0):
        raise ConfigurationError("Adaptive chroma percentile must be in [0, 50).")
    if roi.adaptive_chroma_padding < 0:
        raise ConfigurationError("Adaptive chroma padding must be non-negative.")
    if not (0.0 < roi.temporal_adaptive_alpha <= 1.0):
        raise ConfigurationError("Temporal adaptive alpha must be in (0, 1].")
    if roi.temporal_adaptive_min_updates < 1:
        raise ConfigurationError("Temporal adaptive minimum updates must be at least one.")
    if not (0.0 <= roi.robust_mean_low_percentile < roi.robust_mean_high_percentile <= 100.0):
        raise ConfigurationError("Robust mean percentiles must satisfy 0 <= low < high <= 100.")
    if not (0.0 < roi.coordinate_smoothing_factor <= 1.0):
        raise ConfigurationError("ROI coordinate smoothing factor must be in (0, 1].")
    if roi.max_landmark_motion <= 0.0:
        raise ConfigurationError("Maximum landmark motion must be positive.")
    if roi.face_loss_reset_sec <= 0.0:
        raise ConfigurationError("Face-loss reset timeout must be positive.")
    for name, value in (
        ("face detection confidence", roi.min_face_detection_confidence),
        ("face presence confidence", roi.min_face_presence_confidence),
        ("tracking confidence", roi.min_tracking_confidence),
    ):
        if not (0.0 <= value <= 1.0):
            raise ConfigurationError(f"Minimum {name} must be in [0, 1].")

    analysis = cfg.analysis
    if not (0.0 < analysis.min_accepted_bpm < analysis.max_accepted_bpm):
        raise ConfigurationError("BPM limits require 0 < min_accepted_bpm < max_accepted_bpm.")
    if analysis.analysis_window_sec <= 0.0:
        raise ConfigurationError("Analysis window length must be positive.")
    if not (0.0 <= analysis.min_warmup_sec < analysis.analysis_window_sec):
        raise ConfigurationError("Warmup length must be non-negative and shorter than the analysis window.")
    if analysis.analysis_update_interval_sec <= 0.0:
        raise ConfigurationError("Analysis update interval must be positive.")
    if not (1 <= analysis.butterworth_order <= 10):
        raise ConfigurationError("Butterworth order should be between 1 and 10.")
    if analysis.filter_padtype not in {"odd", "even", "constant"}:
        raise ConfigurationError("Filter padtype must be odd, even, or constant.")
    if analysis.filter_transient_trim_sec < 0.0:
        raise ConfigurationError("Filter transient trim must be non-negative.")
    if not (0.0 < analysis.bpm_smoothing_factor <= 1.0):
        raise ConfigurationError("BPM smoothing factor must be in (0, 1].")
    if analysis.bpm_smoothing_time_constant_sec <= 0.0:
        raise ConfigurationError("BPM smoothing time constant must be positive.")
    if analysis.bpm_history_sec <= 0.0:
        raise ConfigurationError("BPM history length must be positive.")
    if analysis.smoothness_priors_lambda <= 0.0:
        raise ConfigurationError("Smoothness-priors lambda must be positive.")
    if analysis.smoothness_priors_reference_fps <= 0.0:
        raise ConfigurationError("Smoothness-priors reference FPS must be positive.")
    if analysis.min_spectral_snr < 0.0:
        raise ConfigurationError("Minimum spectral SNR must be non-negative.")
    if analysis.max_bpm_jump_per_update <= 0.0:
        raise ConfigurationError("Maximum BPM jump must be positive.")
    if analysis.max_consecutive_low_quality_rejections < 1:
        raise ConfigurationError("Maximum consecutive low-quality rejections must be at least one.")
    if analysis.adaptive_half_bandwidth_hz <= 0.0:
        raise ConfigurationError("Adaptive half-bandwidth must be positive.")
    if analysis.min_adaptive_bandwidth_hz <= 0.0:
        raise ConfigurationError("Minimum adaptive bandwidth must be positive.")
    if analysis.min_effective_fps <= 0.0:
        raise ConfigurationError("Minimum effective FPS must be positive.")
    if analysis.min_uniform_samples < 2:
        raise ConfigurationError("Minimum uniform samples must be at least 2.")
    if analysis.min_interpolation_samples < 2:
        raise ConfigurationError("Minimum interpolation samples must be at least 2.")
    if analysis.candidate_projection_window_sec <= 0.0:
        raise ConfigurationError("Candidate projection window length must be positive.")
    if analysis.min_band_low_hz <= 0.0 or analysis.nyquist_high_fraction <= 0.0:
        raise ConfigurationError("Frequency-band guard settings must be positive.")
    if analysis.welch_window_sec <= 0.0:
        raise ConfigurationError("Welch window length must be positive.")
    if analysis.welch_min_segment_samples < 2:
        raise ConfigurationError("Welch minimum segment samples must be at least 2.")
    if analysis.welch_target_resolution_hz <= 0.0:
        raise ConfigurationError("Welch target resolution must be positive.")
    if analysis.peak_prominence_std_fraction < 0.0 or analysis.min_peak_prominence < 0.0:
        raise ConfigurationError("Peak prominence settings must be non-negative.")
    if analysis.peak_match_tolerance_bpm <= 0.0:
        raise ConfigurationError("Peak match tolerance must be positive.")
    if not (0.0 <= analysis.peak_fusion_fft_weight <= 1.0):
        raise ConfigurationError("FFT fusion weight must be in [0, 1].")
    if not (0.0 <= analysis.peak_fusion_peak_weight <= 1.0):
        raise ConfigurationError("Peak fusion weight must be in [0, 1].")
    if analysis.peak_fusion_fft_weight + analysis.peak_fusion_peak_weight <= 0.0:
        raise ConfigurationError("At least one configured fusion weight must be positive.")
    if analysis.spectral_peak_exclusion_hz <= 0.0:
        raise ConfigurationError("Spectral peak exclusion width must be positive.")
    if not (0.0 < analysis.spectral_edge_penalty <= 1.0):
        raise ConfigurationError("Spectral edge penalty must be in (0, 1].")
    if analysis.spectral_edge_penalty_hz < 0.0:
        raise ConfigurationError("Spectral edge penalty width must be non-negative.")
    if not (0.0 <= analysis.harmonic_power_fraction <= 1.0):
        raise ConfigurationError("Harmonic power fraction must be in [0, 1].")
    if analysis.subharmonic_tolerance_hz <= 0.0 or analysis.harmonic_tolerance_hz <= 0.0:
        raise ConfigurationError("Harmonic tolerances must be positive.")
    if analysis.harmonic_quality_bonus < 1.0:
        raise ConfigurationError("Harmonic quality bonus must be at least 1.")
    if analysis.pca_min_component_std < 0.0:
        raise ConfigurationError("PCA minimum component std must be non-negative.")
    if analysis.pca_sign_min_abs_correlation < 0.0:
        raise ConfigurationError("PCA sign correlation threshold must be non-negative.")
    if analysis.pca_sign_min_samples < 2:
        raise ConfigurationError("PCA sign alignment requires at least two samples.")
    if analysis.kalman_initial_bpm <= 0.0:
        raise ConfigurationError("Kalman initial BPM must be positive.")
    if analysis.kalman_initial_variance <= 0.0:
        raise ConfigurationError("Kalman initial variance must be positive.")
    if analysis.kalman_process_bpm_variance < 0.0 or analysis.kalman_process_rate_variance < 0.0:
        raise ConfigurationError("Kalman process variances must be non-negative.")
    if analysis.kalman_measurement_variance <= 0.0:
        raise ConfigurationError("Kalman measurement variance must be positive.")
    if analysis.global_band_probe_min_quality_ratio < 1.0:
        raise ConfigurationError("Global-band probe quality ratio must be at least 1.")
    if analysis.global_band_probe_interval_updates < 1:
        raise ConfigurationError("Global-band probe interval must be at least one update.")
    if analysis.max_timestamp_gap_factor <= 1.0:
        raise ConfigurationError("Maximum timestamp gap factor must be greater than 1.")
    if not (0.0 < analysis.min_sample_coverage_fraction <= 1.0):
        raise ConfigurationError("Minimum sample coverage must be in (0, 1].")
    if analysis.max_sample_age_sec <= 0.0:
        raise ConfigurationError("Maximum sample age must be positive.")
    if analysis.max_interpolation_gap_factor <= 1.0:
        raise ConfigurationError("Maximum interpolation gap factor must be greater than 1.")
    if analysis.max_illumination_instability <= 0.0:
        raise ConfigurationError("Maximum illumination instability must be positive.")
    allowed_methods = {"all", "select all", "auto", "best", "chrom", "pos", "green", "pca", "ica"}
    if analysis.extraction_method.lower() not in allowed_methods:
        raise ConfigurationError(f"Extraction method must be one of {sorted(allowed_methods)}.")


CFG = default_config()
