from dataclasses import replace
from pathlib import Path

import pytest

from rppg.app.cli import choose_input_source, config_from_args
from rppg.config import default_config, parse_camera_source, validate_config, with_overrides
from rppg.exceptions import ConfigurationError


def test_default_config_validates():
    validate_config(default_config())


def test_parse_numeric_camera_source():
    assert parse_camera_source("0") == 0


def test_parse_empty_camera_source_rejected():
    with pytest.raises(ConfigurationError, match="must not be empty"):
        parse_camera_source(" ")


def test_invalid_roi_mode_rejected():
    cfg = with_overrides(default_config(), roi_mode="nose")
    with pytest.raises(ConfigurationError, match="ROI mode"):
        validate_config(cfg)


def test_missing_video_source_rejected(tmp_path: Path):
    cfg = with_overrides(default_config(), source=str(tmp_path / "missing.mp4"))
    with pytest.raises(ConfigurationError, match="does not exist"):
        validate_config(cfg)


def test_warmup_must_be_shorter_than_window():
    cfg = default_config()
    cfg = replace(
        cfg,
        analysis=replace(
            cfg.analysis,
            min_warmup_sec=cfg.analysis.analysis_window_sec,
        ),
    )
    with pytest.raises(ConfigurationError, match="Warmup length"):
        validate_config(cfg)


def test_frequency_guard_boundaries_rejected():
    cfg = default_config()
    cfg = replace(
        cfg,
        analysis=replace(
            cfg.analysis,
            nyquist_high_fraction=0.0,
        ),
    )
    with pytest.raises(ConfigurationError, match="Frequency-band guard"):
        validate_config(cfg)


def test_invalid_timestamp_guard_rejected():
    cfg = default_config()
    cfg = replace(cfg, analysis=replace(cfg.analysis, max_timestamp_gap_factor=1.0))
    with pytest.raises(ConfigurationError, match="timestamp gap"):
        validate_config(cfg)


def test_invalid_candidate_projection_window_rejected():
    cfg = default_config()
    cfg = replace(cfg, analysis=replace(cfg.analysis, candidate_projection_window_sec=0.0))
    with pytest.raises(ConfigurationError, match="projection window"):
        validate_config(cfg)


def test_invalid_low_quality_hysteresis_rejected():
    cfg = default_config()
    cfg = replace(cfg, analysis=replace(cfg.analysis, max_consecutive_low_quality_rejections=0))
    with pytest.raises(ConfigurationError, match="consecutive low-quality"):
        validate_config(cfg)


def test_zero_total_fusion_weight_rejected():
    cfg = default_config()
    cfg = replace(cfg, analysis=replace(cfg.analysis, peak_fusion_fft_weight=0.0, peak_fusion_peak_weight=0.0))
    with pytest.raises(ConfigurationError, match="fusion weight"):
        validate_config(cfg)


def test_invalid_global_band_probe_interval_rejected():
    cfg = default_config()
    cfg = replace(cfg, analysis=replace(cfg.analysis, global_band_probe_interval_updates=0))
    with pytest.raises(ConfigurationError, match="Global-band probe interval"):
        validate_config(cfg)


def test_benchmark_override_enables_headless_mode():
    cfg = with_overrides(default_config(), benchmark=True)
    assert cfg.benchmark is True
    assert cfg.headless is True


def test_adaptive_skin_strategy_is_an_explicit_override():
    cfg = with_overrides(default_config(), skin_mask_strategy="adaptive_chroma")
    assert cfg.roi.skin_mask_strategy == "adaptive_chroma"


def test_cli_exposes_analysis_and_projection_window_overrides():
    cfg = config_from_args(
        [
            "--analysis-window-seconds",
            "30",
            "--warmup-seconds",
            "20",
            "--candidate-projection-window-seconds",
            "2.0",
            "--global-band-probe-interval-updates",
            "4",
        ]
    )

    assert cfg.analysis.analysis_window_sec == pytest.approx(30.0)
    assert cfg.analysis.min_warmup_sec == pytest.approx(20.0)
    assert cfg.analysis.candidate_projection_window_sec == pytest.approx(2.0)
    assert cfg.analysis.global_band_probe_interval_updates == 4


def test_input_selector_uses_camera_for_live_choice():
    assert choose_input_source(lambda _prompt: "1") == "0"


def test_input_selector_uses_file_chooser_for_saved_video():
    prompts = iter(["2"])
    assert choose_input_source(lambda _prompt: next(prompts), lambda _input: r"C:\\videos\\clip.mp4") == r"C:\\videos\\clip.mp4"


def test_input_selector_returns_none_when_file_selection_is_cancelled():
    assert choose_input_source(lambda _prompt: "saved", lambda _input: None) is None
