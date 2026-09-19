"""Tests for the desktop GUI components, theme, widgets, and engine."""

from __future__ import annotations

import numpy as np
import pytest

from rppg.gui.app_window import RppgAppWindow
from rppg.gui.engine import GuiTelemetry, RppgEngine
from rppg.gui.theme import FONTS, PALETTE, apply_custom_theme, get_theme, set_theme
from rppg.gui.widgets import (
    BpmComparisonCanvas,
    MetricCard,
    SpectrumCanvas,
    StatusBadge,
    VideoCanvas,
    WaveformCanvas,
)


@pytest.fixture(scope="module")
def app_instance():
    """Create a single shared hidden RppgAppWindow for the entire GUI test module."""
    app = RppgAppWindow()
    app.withdraw()  # Hide from screen during test runs
    yield app
    try:
        app.destroy()
    except Exception:
        pass


def test_theme_palette_and_fonts():
    """Verify essential theme tokens and fonts exist."""
    required_colors = [
        "bg_root",
        "bg_panel",
        "bg_card",
        "text_primary",
        "accent_heart",
        "accent_emerald",
        "accent_cyan",
    ]
    for col in required_colors:
        assert col in PALETTE
        assert PALETTE[col].startswith("#")

    required_fonts = ["title", "heading", "body", "bpm_large", "caption"]
    for font_name in required_fonts:
        assert font_name in FONTS
        family, size, *_ = FONTS[font_name]
        assert isinstance(family, str)
        assert isinstance(size, int) and size > 0


def test_apply_custom_theme(app_instance):
    """Verify custom theme registers styles cleanly."""
    style = apply_custom_theme(app_instance)
    assert style is not None


def test_theme_palette_switches_in_place(app_instance):
    """Verify both appearances have complete tokens and can be reapplied."""
    original = get_theme()
    try:
        set_theme("light")
        assert get_theme() == "light"
        assert PALETTE["bg_root"] == "#EEF5FC"
        assert apply_custom_theme(app_instance) is not None

        set_theme("dark")
        assert PALETTE["bg_root"] == "#050B14"
    finally:
        set_theme(original)
        apply_custom_theme(app_instance)


def test_status_badge(app_instance):
    """Verify StatusBadge initialization and state changes."""
    badge = StatusBadge(app_instance)
    badge.pack()
    app_instance.update_idletasks()
    assert badge is not None

    badge.set_status("MONITORING", PALETTE["accent_emerald"])
    assert badge._state_text == "MONITORING"
    assert badge._color == PALETTE["accent_emerald"]

    badge.set_status("WARMUP")
    assert badge._state_text == "WARMUP"
    assert badge._color == PALETTE["accent_amber"]

    # The status pill must render at its requested dimensions before Tk has
    # completed layout, preventing the initial READY label from drifting.
    assert badge._width == 116
    assert badge._height == 30

    badge.destroy()


def test_metric_card(app_instance):
    """Verify MetricCard rendering and value updates."""
    card = MetricCard(app_instance, title="Heart Rate", value="75", unit="BPM", large=True)
    card.pack()
    app_instance.update_idletasks()

    assert card.lbl_title.cget("text") == "HEART RATE"
    assert card.lbl_value.cget("text") == "75"
    assert card.lbl_unit.cget("text") == " BPM"

    card.set_value("80", hint="Stable rhythm", color="#10B981")
    assert card.lbl_value.cget("text") == "80"
    assert card.lbl_hint.cget("text") == "Stable rhythm"

    card.destroy()


def test_video_canvas(app_instance):
    """Verify VideoCanvas placeholder and frame update handling."""
    canvas = VideoCanvas(app_instance, width=200, height=150)
    canvas.pack()
    app_instance.update_idletasks()

    # Create synthetic BGR frame
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_frame[:, :] = (120, 180, 240)
    canvas.update_frame(dummy_frame)
    assert canvas._photo_image is not None

    canvas.clear()
    assert canvas._photo_image is None
    canvas.destroy()


def test_waveform_canvas(app_instance):
    """Verify WaveformCanvas drawing and peak highlighting."""
    canvas = WaveformCanvas(app_instance, height=100)
    canvas.pack()
    app_instance.update_idletasks()

    # Feed synthetic sine wave signal
    t = np.linspace(0, 4 * np.pi, 100)
    signal = np.sin(t)
    peaks = np.array([25, 75])

    canvas.update_signal(signal, peaks)
    assert canvas._data is not None
    assert len(canvas._peaks) == 2

    canvas.clear()
    assert canvas._data is None
    canvas.destroy()


def test_spectrum_canvas(app_instance):
    """Verify SpectrumCanvas Welch PSD rendering."""
    canvas = SpectrumCanvas(app_instance, height=100)
    canvas.pack()
    app_instance.update_idletasks()

    freqs = np.linspace(0.5, 4.0, 50)
    power = np.exp(-((freqs - 1.2) ** 2) / 0.1)
    canvas.update_spectrum(freqs, power, dom_freq=1.2)

    assert canvas._freqs is not None
    assert canvas._dom_freq == 1.2

    canvas.clear()
    assert canvas._freqs is None
    canvas.destroy()


def test_bpm_comparison_canvas(app_instance):
    """Verify BpmComparisonCanvas renders trend lines and ground truth comparisons."""
    canvas = BpmComparisonCanvas(app_instance, width=200, height=150)
    canvas.pack()
    app_instance.update_idletasks()

    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    achieved = [72.0, 73.5, 74.0, 72.8, 73.0]
    ground_truths = [71.0, 72.0, 73.0, 74.0, 73.5]

    canvas.update_data(times, estimates=achieved, current_time=5.0, ground_truths=ground_truths, has_ground_truth=True)
    assert len(canvas._times) == 5
    assert len(canvas._estimates) == 5
    assert len(canvas._references) == 5
    displayed_labels = [
        canvas.itemcget(item, "text")
        for item in canvas.find_all()
        if canvas.type(item) == "text"
    ]
    assert "Estimate" in displayed_labels
    assert "Reference" in displayed_labels
    item_types = {canvas.type(item) for item in canvas.find_all()}
    assert "oval" not in item_types
    assert "rectangle" not in item_types

    canvas.clear()
    assert len(canvas._times) == 0
    assert len(canvas._estimates) == 0
    canvas.destroy()


def test_engine_lifecycle():
    """Verify RppgEngine initialization, state check, and queue draining."""
    engine = RppgEngine()
    assert not engine.is_running
    assert engine.get_frame() is None
    assert engine.get_events() == []

    # Stop when not running should safely no-op
    engine.stop()
    assert not engine.is_running


def test_app_window_creation(app_instance):
    """Verify RppgAppWindow builds full UI structure without error."""
    assert app_instance.video_canvas is not None
    assert app_instance.comparison_canvas is not None
    assert app_instance.card_bpm is not None
    assert app_instance.card_fft is not None
    assert app_instance.card_peaks is not None
    assert app_instance.card_sqi is not None
    assert app_instance.card_ground_truth is not None
    assert app_instance.waveform_canvas is not None
    assert app_instance.spectrum_canvas is not None
    assert app_instance.card_sqi.grid_info()["column"] == 0
    assert app_instance.card_bpm.grid_info()["column"] == 3
    assert app_instance.cbo_method is not None
    assert app_instance.btn_theme is not None
    assert len(app_instance.source_options) == 3
    assert app_instance.source_options[0]._variable.get() == "camera"
    assert "Select All (Auto-Best SQI)" in app_instance.cbo_method["values"]
    assert "CHROM (Chrominance)" in app_instance.cbo_method["values"]
    assert "POS (Plane-Orthogonal-to-Skin)" in app_instance.cbo_method["values"]
    assert "GREEN (Green Channel)" in app_instance.cbo_method["values"]

    # Test applying telemetry payload
    telemetry = GuiTelemetry(
        status_text="72.0 BPM",
        face_detected=True,
        smoothed_bpm=72.0,
        quality=0.92,
        bpm_fft=72.0,
        bpm_peaks=71.8,
        filtered_signal=np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
        peaks=np.array([2]),
        freqs=np.array([1.0, 1.2, 1.4]),
        power=np.array([0.1, 1.0, 0.2]),
        dom_freq=1.2,
        effective_fps=30.0,
        total_frames=100,
        accepted_samples=95,
        ground_truth_bpm=71.5,
        bpm_history_times=[1.0, 2.0, 3.0],
        bpm_history_values=[70.0, 71.0, 72.0],
        gt_history_values=[71.0, 71.5, 71.5],
        has_ground_truth=True,
        active_method="CHROM",
        skin_pixels=1875,
        landmark_motion=0.009,
    )
    app_instance._apply_telemetry(telemetry)

    assert app_instance.card_bpm.lbl_value.cget("text") == "72.0"
    assert app_instance.card_sqi.lbl_value.cget("text") == "0.92"
    assert "CHROM" in app_instance.card_sqi.lbl_hint.cget("text")
    assert app_instance.card_sqi.note_labels[0].cget("text") == "• Stable facial signal"
    assert app_instance.card_sqi.note_labels[1].cget("text") == "• Low motion interference"
    assert app_instance.card_ground_truth.lbl_value.cget("text") == "71.5"
    assert app_instance.title() == "Pulse Monitor"

    # Test reset functionality
    app_instance._reset_to_defaults()
    assert app_instance.card_bpm.lbl_value.cget("text") == "--"
    assert app_instance.var_source_mode.get() == "camera"
    assert app_instance.var_camera_idx.get() == "0"
    assert app_instance.var_extraction_method.get() == "Select All (Auto-Best SQI)"
    assert app_instance.var_roi_mode.get() == "multi"
    assert app_instance.var_skin_strategy.get() == "fixed"
    assert app_instance.badge_status._state_text == "READY"


def test_source_selector_has_one_selected_option(app_instance):
    """Only the selected input source receives the filled radio indicator."""
    app_instance.var_source_mode.set("video")
    app_instance.update_idletasks()
    selected = [option._variable.get() == option._value for option in app_instance.source_options]
    assert selected == [False, True, False]
    app_instance.var_source_mode.set("camera")


def test_app_theme_toggle_rebuilds_dashboard(app_instance):
    """Verify the header option switches the entire dashboard appearance."""
    initial = app_instance.var_theme.get()
    app_instance._toggle_theme()
    assert app_instance.var_theme.get() != initial
    assert app_instance.btn_theme is not None
    assert app_instance.video_canvas is not None
    app_instance._toggle_theme()
    assert app_instance.var_theme.get() == initial
