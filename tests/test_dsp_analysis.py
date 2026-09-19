from dataclasses import replace

import numpy as np
import pytest

from rppg.algorithms.analysis import (
    _candidate_search_bands,
    analyze_window,
    fuse_estimates_detailed,
    multi_feature_sqi,
    select_best_candidate,
)
from rppg.algorithms.candidates import CandidateSignal, generate_candidate_signals
from rppg.config import AnalysisConfig
from rppg.dsp.filtering import bandpass_filter, clamp_band_to_nyquist
from rppg.dsp.peaks import estimate_peaks_bpm, estimate_peaks_bpm_detailed
from rppg.dsp.preprocessing import zscore_safe
from rppg.dsp.spectral import harmonic_adjusted_frequency, spectral_edge_penalty_multiplier, spectral_quality_and_bpm
from rppg.exceptions import AnalysisError, LowQualitySignalError, WarmupError
from rppg.results import AnalysisResult


def test_zscore_safe_constant_returns_zeros():
    assert np.allclose(zscore_safe(np.ones(5)), np.zeros(5))


def test_clamp_band_to_nyquist_rejects_low_fps():
    with pytest.raises(LowQualitySignalError, match="too low"):
        clamp_band_to_nyquist([1.0, 4.0], 2.0, AnalysisConfig())


def test_analyze_window_rejects_empty_window_as_warmup():
    with pytest.raises(WarmupError, match="No samples"):
        analyze_window([], [], None, AnalysisConfig())


def test_analyze_window_rejects_bad_rgb_shape_as_analysis_error():
    with pytest.raises(AnalysisError, match="Invalid RGB"):
        analyze_window([0.0, 1.0], [1.0, 2.0], None, AnalysisConfig())


def test_analysis_result_legacy_schema():
    result = AnalysisResult(
        name="Green",
        quality=2.5,
        bpm_fft=72.0,
        bpm_peaks=73.0,
        fused_bpm=72.3,
        filtered=np.array([0.0, 1.0], dtype=np.float64),
        peaks=np.array([1], dtype=int),
        freqs=np.array([1.2], dtype=np.float64),
        power=np.array([3.4], dtype=np.float64),
        uniform_times=np.array([0.0, 0.1], dtype=np.float64),
        fps=10.0,
        band=(0.9, 3.3),
    )

    legacy = result.as_dict()
    assert legacy["name"] == "Green"
    assert legacy["fused_bpm"] == pytest.approx(72.3)
    assert np.array_equal(legacy["peaks"], np.array([1]))
    assert legacy["candidate_metadata"] == {}


def test_candidate_interface_includes_green_pca_chrom_and_pos():
    fps = 30.0
    t = np.arange(240, dtype=np.float64) / fps
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    rgb = np.column_stack((0.6 * pulse, pulse, 0.3 * pulse))

    candidates = generate_candidate_signals(rgb, fps, [0.9, 2.0], AnalysisConfig())
    names = {candidate.name for candidate in candidates}

    assert {"Green", "PCA", "CHROM", "POS"}.issubset(names)
    assert all(isinstance(candidate, CandidateSignal) for candidate in candidates)
    assert all(candidate.signal.shape == (len(t),) for candidate in candidates)
    assert all(isinstance(candidate.metadata, dict) for candidate in candidates)
    assert next(candidate for candidate in candidates if candidate.name == "CHROM").metadata["projection_window_count"] > 1
    assert next(candidate for candidate in candidates if candidate.name == "POS").metadata["projection"] == "overlapping_local_alpha"


def test_chrom_and_pos_use_window_normalized_raw_rgb_when_available():
    fps = 30.0
    t = np.arange(240, dtype=np.float64) / fps
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    rgb_norm = np.column_stack((0.6 * pulse, pulse, 0.3 * pulse))
    raw_rgb = np.column_stack((100.0 + 0.8 * pulse, 110.0 + 1.5 * pulse, 90.0 + 0.35 * pulse))

    candidates = generate_candidate_signals(
        rgb_norm,
        fps,
        [0.9, 2.0],
        AnalysisConfig(),
        rgb_for_projection=raw_rgb,
    )

    for name in ("CHROM", "POS"):
        candidate = next(item for item in candidates if item.name == name)
        assert candidate.metadata["input_normalization"] == "window_channel_mean"
        assert candidate.metadata["projection_taper"] == "hann_with_0.1_floor"
        assert np.std(candidate.signal) > 0.0


def test_spectral_peak_refinement_recovers_between_bin_frequency():
    fps = 30.0
    true_hz = 1.23
    t = np.arange(20 * int(fps), dtype=np.float64) / fps
    x = np.sin(2.0 * np.pi * true_hz * t)
    cfg = replace(AnalysisConfig(), min_spectral_snr=0.0, welch_target_resolution_hz=0.12)

    quality, bpm, freqs, power = spectral_quality_and_bpm(x, fps, [0.9, 1.8], cfg)
    raw_peak = freqs[np.argmax(power[(freqs >= 0.9) & (freqs <= 1.8)]) + np.flatnonzero((freqs >= 0.9) & (freqs <= 1.8))[0]]

    assert quality > 1.0
    assert abs((bpm / 60.0) - true_hz) <= abs(raw_peak - true_hz) + 1e-6


def test_spectral_refinement_cannot_escape_the_configured_search_band():
    fps = 30.0
    t = np.arange(20 * int(fps), dtype=np.float64) / fps
    # An out-of-band source can leak power into the lower edge of the search
    # band; refinement must not turn that leakage into a below-band estimate.
    x = np.sin(2.0 * np.pi * 0.82 * t)

    _, bpm, _, _ = spectral_quality_and_bpm(x, fps, [0.9, 1.8], AnalysisConfig())

    assert 54.0 <= bpm <= 108.0


@pytest.mark.parametrize("order", [2, 3, 4])
def test_butterworth_orders_preserve_pulse_band_and_suppress_high_frequency_noise(order: int):
    fps = 30.0
    t = np.arange(30 * int(fps), dtype=np.float64) / fps
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    noise = 0.8 * np.sin(2.0 * np.pi * 6.0 * t)
    filtered = bandpass_filter(pulse + noise, fps, [0.9, 2.0], replace(AnalysisConfig(), butterworth_order=order))

    pulse_correlation = abs(float(np.corrcoef(filtered, pulse)[0, 1]))
    noise_correlation = abs(float(np.corrcoef(filtered, noise)[0, 1]))
    assert pulse_correlation > 0.9
    assert noise_correlation < 0.1


def test_harmonic_adjustment_prefers_supported_fundamental():
    freqs = np.linspace(0.5, 4.0, 351)
    power = np.ones_like(freqs) * 0.01
    power[np.argmin(np.abs(freqs - 1.1))] = 0.7
    power[np.argmin(np.abs(freqs - 2.2))] = 1.0

    adjusted, bonus = harmonic_adjusted_frequency(freqs, power, 2.2, [0.8, 3.0], AnalysisConfig())

    assert adjusted == pytest.approx(1.1, abs=0.03)
    assert bonus >= 1.0


def test_spectral_edge_penalty_is_symmetric():
    cfg = AnalysisConfig(spectral_edge_penalty_hz=0.1, spectral_edge_penalty=0.5)

    assert spectral_edge_penalty_multiplier(0.95, [0.9, 2.0], cfg) == pytest.approx(0.5)
    assert spectral_edge_penalty_multiplier(1.95, [0.9, 2.0], cfg) == pytest.approx(0.5)
    assert spectral_edge_penalty_multiplier(1.45, [0.9, 2.0], cfg) == pytest.approx(1.0)


def test_global_frequency_probe_runs_only_at_its_configured_interval():
    cfg = replace(AnalysisConfig(), global_band_probe_interval_updates=3)

    first = _candidate_search_bands(72.0, 30.0, cfg, analysis_sequence=0)
    middle = _candidate_search_bands(72.0, 30.0, cfg, analysis_sequence=1)
    periodic = _candidate_search_bands(72.0, 30.0, cfg, analysis_sequence=3)

    assert [mode for _, mode in first] == ["adaptive", "global_periodic"]
    assert [mode for _, mode in middle] == ["adaptive"]
    assert [mode for _, mode in periodic] == ["adaptive", "global_periodic"]


def test_multi_feature_sqi_penalizes_peak_disagreement_and_exposes_features():
    agreeing, features = multi_feature_sqi(4.0, 72.0, 73.0, 10, AnalysisConfig())
    disagreeing, _ = multi_feature_sqi(4.0, 72.0, 120.0, 10, AnalysisConfig())

    assert agreeing > disagreeing
    assert features["peak_count"] == 10
    assert features["peak_agreement"] > 0.9


def test_multi_feature_sqi_incorporates_acquisition_and_roi_quality_context():
    baseline, _ = multi_feature_sqi(4.0, 72.0, 72.0, 10, AnalysisConfig())
    degraded, features = multi_feature_sqi(
        4.0,
        72.0,
        72.0,
        10,
        AnalysisConfig(),
        context={
            "sample_coverage": 0.7,
            "detection_availability": 0.6,
            "mask_coverage": 0.5,
            "motion_quality": 0.4,
            "illumination_quality": 0.8,
        },
    )

    assert degraded < baseline
    assert features["acquisition_quality"] == pytest.approx(0.6)
    assert features["mask_coverage"] == pytest.approx(0.5)


def test_fusion_reduces_peak_influence_continuously_when_estimates_disagree():
    cfg = AnalysisConfig(min_spectral_snr=0.0, peak_fusion_peak_weight=0.4)
    agreeing_bpm, agreeing = fuse_estimates_detailed(72.0, 73.0, 4.0, None, cfg)
    disagreeing_bpm, disagreeing = fuse_estimates_detailed(72.0, 110.0, 4.0, None, cfg)

    assert 72.0 < agreeing_bpm < 73.0
    assert agreeing["fusion_peak_weight"] > disagreeing["fusion_peak_weight"] > 0.0
    assert disagreeing_bpm < 80.0
    assert agreeing["fusion_fft_weight"] + agreeing["fusion_peak_weight"] == pytest.approx(1.0)


def test_fusion_uses_the_configured_fft_and_peak_balance_as_its_peak_cap():
    cfg = AnalysisConfig(min_spectral_snr=0.0, peak_fusion_fft_weight=0.2, peak_fusion_peak_weight=0.8)

    _, features = fuse_estimates_detailed(72.0, 72.0, 4.0, None, cfg)

    assert features["fusion_peak_weight_cap"] == pytest.approx(0.8)
    assert features["fusion_peak_weight"] == pytest.approx(0.8)


def test_peak_detector_handles_inverted_polarity():
    fps = 30.0
    t = np.arange(12 * int(fps), dtype=np.float64) / fps
    inverted = -np.sin(2.0 * np.pi * 1.25 * t)

    bpm, peaks = estimate_peaks_bpm(inverted, t, fps, [0.9, 1.8], AnalysisConfig(), allow_inverted=True)

    assert bpm == pytest.approx(75.0, abs=3.0)
    assert len(peaks) >= 5


def test_detailed_peak_estimate_records_the_selected_polarity():
    fps = 30.0
    t = np.arange(12 * int(fps), dtype=np.float64) / fps
    inverted = -np.sin(2.0 * np.pi * 1.25 * t)

    estimate = estimate_peaks_bpm_detailed(inverted, t, fps, [0.9, 1.8], AnalysisConfig(), allow_inverted=True)

    assert estimate.bpm == pytest.approx(75.0, abs=3.0)
    assert estimate.polarity in {"normal", "inverted"}


def test_select_best_candidate_rejects_all_nan_input_before_spectral_analysis():
    cfg = replace(AnalysisConfig(), min_uniform_samples=5)
    uniform_times = np.arange(10, dtype=np.float64) / 10.0
    rgb_norm = np.full((10, 3), np.nan, dtype=np.float64)

    with pytest.raises(LowQualitySignalError, match="Rejected low-quality"):
        select_best_candidate(uniform_times, rgb_norm, 10.0, None, cfg)


def test_filter_transient_trim_keeps_result_arrays_aligned():
    fps = 30.0
    t = np.arange(20 * int(fps), dtype=np.float64) / fps
    pulse = np.sin(2.0 * np.pi * 1.2 * t)
    rgb_norm = np.column_stack((0.4 * pulse, pulse, 0.2 * pulse))
    cfg = replace(
        AnalysisConfig(),
        min_spectral_snr=0.0,
        filter_transient_trim_sec=1.0,
        global_band_probe_enabled=False,
    )

    result = select_best_candidate(t, rgb_norm, fps, None, cfg)

    assert len(result.filtered) == len(result.uniform_times)
    assert len(result.filtered) == len(t) - (2 * int(fps))
    assert result.fused_bpm == pytest.approx(72.0, abs=5.0)


def test_fps_aware_detrending_path_accepts_synthetic_window():
    fps = 30.0
    duration = 24.0
    times = np.arange(int(duration * fps), dtype=np.float64) / fps
    pulse = np.sin(2.0 * np.pi * 1.2 * times)
    trend = 0.01 * times
    rgb = np.column_stack((100.0 + trend + 0.3 * pulse, 110.0 + trend + pulse, 90.0 + trend + 0.2 * pulse))
    cfg = replace(
        AnalysisConfig(),
        min_spectral_snr=0.0,
        max_illumination_instability=1.0,
        global_band_probe_enabled=False,
    )

    result = analyze_window(times, rgb, None, cfg)

    assert result.name in {"Green", "PCA", "CHROM", "POS"}
    assert result.fused_bpm == pytest.approx(72.0, abs=5.0)
    assert result.candidate_metadata
    assert result.quality_features["spectral_score"] > 0.0
