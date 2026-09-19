"""Main desktop application window for the classical rPPG monitor."""

from __future__ import annotations

import os
import platform
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from rppg.config import AppConfig, default_config, with_overrides
from .engine import GuiTelemetry, RppgEngine
from .theme import FONTS, PALETTE, apply_custom_theme, get_theme, set_theme
from .widgets import (
    BpmComparisonCanvas,
    HeartRateHeroWidget,
    IconCanvas,
    MetricCard,
    PulseIconBadge,
    QualityProgressBar,
    SourceRadioButton,
    SpectrumCanvas,
    StatusBadge,
    VideoCanvas,
    WaveformCanvas,
)


class RppgAppWindow(tk.Tk):
    """Modern clinical rPPG monitoring desktop application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Pulse Monitor")
        self.geometry("1260x860")
        self.minsize(1080, 720)
        self.configure(bg=PALETTE["bg_root"])

        self.engine = RppgEngine()
        self.var_theme = tk.StringVar(value=get_theme())
        self._last_telemetry: GuiTelemetry | None = None
        self._last_frame: np.ndarray | None = None
        self.apply_theme()

        # State variables
        self.var_source_mode = tk.StringVar(value="camera")
        self.var_camera_idx = tk.StringVar(value="0")
        self.var_video_path = tk.StringVar(value="")
        self.var_ubfc_path = tk.StringVar(value="")
        self.var_extraction_method = tk.StringVar(value="Select All (Auto-Best SQI)")
        self.var_roi_mode = tk.StringVar(value="multi")
        self.var_skin_strategy = tk.StringVar(value="fixed")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Schedule high-frequency GUI polling loop (~33 FPS)
        self.after(30, self._poll_engine)

    def apply_theme(self) -> None:
        apply_custom_theme(self)

    def _toggle_theme(self) -> None:
        """Switch appearance while preserving source settings and monitoring state."""
        next_theme = "light" if self.var_theme.get() == "dark" else "dark"
        self.var_theme.set(set_theme(next_theme))
        self.configure(bg=PALETTE["bg_root"])
        self.apply_theme()

        # Tk widgets do not automatically inherit new colors. Rebuilding the
        # presentation layer keeps every card, chart, and custom canvas in sync
        # without interrupting the acquisition engine.
        for child in self.winfo_children():
            child.destroy()
        self._build_ui()

        if self._last_frame is not None:
            self.video_canvas.update_frame(self._last_frame)
        if self._last_telemetry is not None:
            self._apply_telemetry(self._last_telemetry)
        elif self.engine.is_running:
            self.badge_status.set_status("MONITORING", PALETTE["accent_emerald"])
            self.lbl_status_left.config(text="Monitoring...")
            self.btn_toggle_monitor.config(text="■ Stop Monitoring", style="Stop.TButton")

    def _build_ui(self) -> None:
        """Construct the modern structured layout matching the reference mockup."""
        # 1. Top Header Bar
        self.header_frame = tk.Frame(self, bg=PALETTE["bg_panel"], height=60, padx=20, pady=8)
        self.header_frame.pack(fill="x", side="top")
        self._build_header(self.header_frame)

        # 2. Bottom Telemetry Statusbar
        self.statusbar_frame = tk.Frame(self, bg=PALETTE["bg_panel"], height=30, padx=18, pady=4)
        self.statusbar_frame.pack(fill="x", side="bottom")
        self._build_statusbar(self.statusbar_frame)

        # 3. Main Workspace Split (Sidebar + Content)
        self.main_container = tk.Frame(self, bg=PALETTE["bg_root"], padx=14, pady=10)
        self.main_container.pack(fill="both", expand=True)

        # Left Sidebar (Controls & Settings)
        self.sidebar_frame = tk.Frame(
            self.main_container,
            bg=PALETTE["bg_panel"],
            width=290,
            padx=14,
            pady=14,
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        self.sidebar_frame.pack(side="left", fill="y", padx=(0, 10))
        self.sidebar_frame.pack_propagate(False)
        self._build_sidebar(self.sidebar_frame)

        # Right Content Area (Video Preview + Heart Rate Analysis + DSP Waveforms + Metric Cards)
        self.content_frame = tk.Frame(self.main_container, bg=PALETTE["bg_root"])
        self.content_frame.pack(side="right", fill="both", expand=True)
        self._build_content(self.content_frame)

    # -------------------------------------------------------------------------
    # Header & Statusbar
    # -------------------------------------------------------------------------
    def _build_header(self, parent: tk.Frame) -> None:
        left_box = tk.Frame(parent, bg=PALETTE["bg_panel"])
        left_box.pack(side="left", fill="y")

        # Circular dark pulse badge with cyan ECG pulse wave
        badge_ico = PulseIconBadge(left_box, size=34)
        badge_ico.pack(side="left", padx=(0, 10))

        title_box = tk.Frame(left_box, bg=PALETTE["bg_panel"])
        title_box.pack(side="left", fill="y")

        lbl_title = tk.Label(
            title_box,
            text="Pulse Monitor",
            font=FONTS["title"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_primary"],
        )
        lbl_title.pack(anchor="w")

        lbl_subtitle = tk.Label(
            title_box,
            text="Remote Photoplethysmography",
            font=FONTS["subtitle"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_secondary"],
        )
        lbl_subtitle.pack(anchor="w")

        # Right controls: status is a compact, stable header anchor.
        right_box = tk.Frame(parent, bg=PALETTE["bg_panel"])
        right_box.pack(side="right", fill="y")

        self.badge_status = StatusBadge(right_box)
        self.badge_status.pack(side="left", padx=(0, 10), pady=4)
        self.badge_status.set_status("READY", PALETTE["text_muted"])

        self.btn_theme = ttk.Button(
            right_box,
            text="Light mode" if self.var_theme.get() == "dark" else "Dark mode",
            style="Theme.TButton",
            command=self._toggle_theme,
        )
        self.btn_theme.pack(side="left", padx=(0, 8))

        self.btn_open_folder = ttk.Button(
            right_box,
            text="Session output",
            style="Secondary.TButton",
            command=self._open_session_dir,
        )
        self.btn_open_folder.pack(side="left")

    def _build_statusbar(self, parent: tk.Frame) -> None:
        left_status_box = tk.Frame(parent, bg=PALETTE["bg_panel"])
        left_status_box.pack(side="left")

        self.lbl_status_dot = tk.Label(
            left_status_box,
            text="●",
            font=FONTS["caption"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["accent_emerald"],
        )
        self.lbl_status_dot.pack(side="left", padx=(0, 4))

        self.lbl_status_left = tk.Label(
            left_status_box,
            text="Ready",
            font=FONTS["caption"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_secondary"],
        )
        self.lbl_status_left.pack(side="left")

        self.lbl_status_right = tk.Label(
            parent,
            text="FPS: 0.0   Frames: 0   Accepted: 0",
            font=FONTS["caption"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_muted"],
        )
        self.lbl_status_right.pack(side="right")

    # -------------------------------------------------------------------------
    # Left Sidebar (Controls & Settings)
    # -------------------------------------------------------------------------
    def _build_sidebar(self, parent: tk.Frame) -> None:
        # Bottom Container: Action Buttons pinned to bottom of sidebar
        btn_box = tk.Frame(parent, bg=PALETTE["bg_panel"])
        btn_box.pack(side="bottom", fill="x", pady=(10, 0))

        self.btn_toggle_monitor = ttk.Button(
            btn_box,
            text="▶ Start Monitoring",
            style="Start.TButton",
            command=self._toggle_monitor,
        )
        self.btn_toggle_monitor.pack(fill="x", pady=(0, 8))

        self.btn_reset = ttk.Button(
            btn_box,
            text="↺ Reset",
            style="Reset.TButton",
            command=self._reset_to_defaults,
        )
        self.btn_reset.pack(fill="x")

        # Top Container: Sections expand gracefully
        top_box = tk.Frame(parent, bg=PALETTE["bg_panel"])
        top_box.pack(side="top", fill="both", expand=True)

        # Section 1: Input Source
        sec1_hdr = tk.Frame(top_box, bg=PALETTE["bg_panel"])
        sec1_hdr.pack(fill="x", pady=(0, 6))

        IconCanvas(sec1_hdr, name="camera", size=14, color=PALETTE["accent_cyan"], bg=PALETTE["bg_panel"]).pack(side="left", padx=(0, 6))

        tk.Label(
            sec1_hdr,
            text="INPUT SOURCE",
            font=FONTS["caption_bold"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_primary"],
        ).pack(side="left")

        modes = [
            ("Live Webcam", "camera"),
            ("Video File", "video"),
            ("UBFC Dataset", "ubfc"),
        ]
        self.source_options: list[SourceRadioButton] = []
        for text, val in modes:
            rb = SourceRadioButton(
                top_box,
                text=text,
                value=val,
                variable=self.var_source_mode,
                command=self._on_source_mode_changed,
            )
            rb.pack(anchor="w", pady=3)
            self.source_options.append(rb)

        # Source-specific parameter container
        self.src_details_frame = tk.Frame(top_box, bg=PALETTE["bg_panel"], pady=4)
        self.src_details_frame.pack(fill="x")
        self._update_source_details_ui()

        ttk.Separator(top_box, orient="horizontal").pack(fill="x", pady=12)

        # Section 2: Method Selection
        sec2_hdr = tk.Frame(top_box, bg=PALETTE["bg_panel"])
        sec2_hdr.pack(fill="x", pady=(0, 6))

        IconCanvas(sec2_hdr, name="grid", size=14, color=PALETTE["accent_cyan"], bg=PALETTE["bg_panel"]).pack(side="left", padx=(0, 6))

        tk.Label(
            sec2_hdr,
            text="METHOD SELECTION",
            font=FONTS["caption_bold"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_primary"],
        ).pack(side="left")

        tk.Label(
            top_box,
            text="Extraction Algorithm",
            font=FONTS["caption"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_secondary"],
        ).pack(anchor="w", pady=(0, 3))

        self.cbo_method = ttk.Combobox(
            top_box,
            textvariable=self.var_extraction_method,
            values=[
                "Select All (Auto-Best SQI)",
                "CHROM (Chrominance)",
                "POS (Plane-Orthogonal-to-Skin)",
                "GREEN (Green Channel)",
                "PCA (Principal Components)",
                "ICA (Independent Components)",
            ],
            state="readonly",
        )
        self.cbo_method.pack(fill="x", pady=(0, 6))

        ttk.Separator(top_box, orient="horizontal").pack(fill="x", pady=12)

        # Section 3: Face Tracking
        sec3_hdr = tk.Frame(top_box, bg=PALETTE["bg_panel"])
        sec3_hdr.pack(fill="x", pady=(0, 4))

        IconCanvas(sec3_hdr, name="target", size=14, color=PALETTE["accent_cyan"], bg=PALETTE["bg_panel"]).pack(side="left", padx=(0, 6))

        tk.Label(
            sec3_hdr,
            text="FACE TRACKING",
            font=FONTS["caption_bold"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_primary"],
        ).pack(side="left")

        # Tracking status indicator
        self.lbl_face_status = tk.Label(
            top_box,
            text="● Inactive",
            font=FONTS["caption_bold"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_muted"],
        )
        self.lbl_face_status.pack(anchor="w", pady=(2, 8))

        # 3 Metrics (FPS, Frames, Accepted Frames) in elevated card
        stats_frame = tk.Frame(
            top_box,
            bg=PALETTE["bg_card_inner"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            padx=10,
            pady=8,
        )
        stats_frame.pack(fill="x", pady=(0, 8))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(2, weight=1)

        # FPS
        f1 = tk.Frame(stats_frame, bg=PALETTE["bg_card_inner"])
        f1.grid(row=0, column=0, sticky="w")
        tk.Label(f1, text="FPS", font=FONTS["caption"], bg=PALETTE["bg_card_inner"], fg=PALETTE["text_secondary"]).pack(anchor="w")
        self.lbl_fps_val = tk.Label(f1, text="--", font=FONTS["heading"], bg=PALETTE["bg_card_inner"], fg=PALETTE["text_primary"])
        self.lbl_fps_val.pack(anchor="w")

        # Frames
        f2 = tk.Frame(stats_frame, bg=PALETTE["bg_card_inner"])
        f2.grid(row=0, column=1, sticky="w")
        tk.Label(f2, text="Frames", font=FONTS["caption"], bg=PALETTE["bg_card_inner"], fg=PALETTE["text_secondary"]).pack(anchor="w")
        self.lbl_frames_val = tk.Label(f2, text="--", font=FONTS["heading"], bg=PALETTE["bg_card_inner"], fg=PALETTE["text_primary"])
        self.lbl_frames_val.pack(anchor="w")

        # Accepted Frames
        f3 = tk.Frame(stats_frame, bg=PALETTE["bg_card_inner"])
        f3.grid(row=0, column=2, sticky="w")
        tk.Label(f3, text="Accepted", font=FONTS["caption"], bg=PALETTE["bg_card_inner"], fg=PALETTE["text_secondary"]).pack(anchor="w")
        self.lbl_acc_frames_val = tk.Label(f3, text="--", font=FONTS["heading"], bg=PALETTE["bg_card_inner"], fg=PALETTE["text_primary"])
        self.lbl_acc_frames_val.pack(anchor="w")


    def _update_source_details_ui(self) -> None:
        """Render source inputs based on selected source mode."""
        for child in self.src_details_frame.winfo_children():
            child.destroy()

        mode = self.var_source_mode.get()
        if mode == "camera":
            self.var_camera_idx.set("0")
        elif mode == "video":
            tk.Label(
                self.src_details_frame,
                text="Video File",
                font=FONTS["caption"],
                bg=PALETTE["bg_panel"],
                fg=PALETTE["text_secondary"],
            ).pack(anchor="w")
            row = tk.Frame(self.src_details_frame, bg=PALETTE["bg_panel"])
            row.pack(fill="x", pady=2)
            ent = tk.Entry(
                row,
                textvariable=self.var_video_path,
                bg=PALETTE["bg_card_inner"],
                fg=PALETTE["text_primary"],
                insertbackground=PALETTE["text_primary"],
                font=FONTS["caption"],
            )
            ent.pack(side="left", fill="x", expand=True, padx=(0, 4))
            btn = ttk.Button(row, text="Browse", width=7, command=self._browse_video_file)
            btn.pack(side="right")
        elif mode == "ubfc":
            tk.Label(
                self.src_details_frame,
                text="Dataset Folder",
                font=FONTS["caption"],
                bg=PALETTE["bg_panel"],
                fg=PALETTE["text_secondary"],
            ).pack(anchor="w")
            row = tk.Frame(self.src_details_frame, bg=PALETTE["bg_panel"])
            row.pack(fill="x", pady=2)
            ent = tk.Entry(
                row,
                textvariable=self.var_ubfc_path,
                bg=PALETTE["bg_card_inner"],
                fg=PALETTE["text_primary"],
                insertbackground=PALETTE["text_primary"],
                font=FONTS["caption"],
            )
            ent.pack(side="left", fill="x", expand=True, padx=(0, 4))
            btn = ttk.Button(row, text="Browse", width=7, command=self._browse_ubfc_folder)
            btn.pack(side="right")

    def _on_source_mode_changed(self) -> None:
        self._update_source_details_ui()

    def _browse_video_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm"),
                ("All Files", "*.*"),
            ],
        )
        if chosen:
            self.var_video_path.set(chosen)

    def _browse_ubfc_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Select UBFC Dataset Subject Folder")
        if chosen:
            self.var_ubfc_path.set(chosen)

    # -------------------------------------------------------------------------
    # Main Content Area (3 Tiers)
    # -------------------------------------------------------------------------
    def _build_content(self, parent: tk.Frame) -> None:
        # Tier 1 (Top Row): Video & ROI (Left) + Heart Rate Analysis (Right)
        top_row = tk.Frame(parent, bg=PALETTE["bg_root"])
        top_row.pack(fill="both", expand=True, pady=(0, 8))
        top_row.columnconfigure(0, weight=1)
        top_row.columnconfigure(1, weight=1)
        top_row.rowconfigure(0, weight=1)

        # 1. Video & ROI Card
        video_card = tk.Frame(
            top_row,
            bg=PALETTE["bg_card"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            padx=12,
            pady=10,
        )
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Video Header
        v_hdr = tk.Frame(video_card, bg=PALETTE["bg_card"])
        v_hdr.pack(fill="x", pady=(0, 6))

        IconCanvas(v_hdr, name="camera", size=14, color=PALETTE["accent_cyan"], bg=PALETTE["bg_card"]).pack(side="left", padx=(0, 6))

        tk.Label(
            v_hdr,
            text="Video & ROI",
            font=FONTS["heading"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_primary"],
        ).pack(side="left")

        # Video Canvas
        self.video_canvas = VideoCanvas(video_card, width=420, height=250)
        self.video_canvas.pack(fill="both", expand=True)

        # Video Footer
        v_foot = tk.Frame(video_card, bg=PALETTE["bg_card"])
        v_foot.pack(fill="x", pady=(6, 0))

        self.lbl_video_face_status = tk.Label(
            v_foot,
            text="● Searching for face...",
            font=FONTS["caption"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["accent_amber"],
        )
        self.lbl_video_face_status.pack(side="left")

        lbl_roi_pill = tk.Label(
            v_foot,
            text="ROI Overlays",
            font=FONTS["caption"],
            bg=PALETTE["bg_card_inner"],
            fg=PALETTE["text_secondary"],
            padx=8,
            pady=2,
            relief="solid",
            borderwidth=1,
        )
        lbl_roi_pill.pack(side="right")

        # 2. Heart Rate Analysis Card
        hr_card = tk.Frame(
            top_row,
            bg=PALETTE["bg_card"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            padx=12,
            pady=10,
        )
        hr_card.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # HR Header with BPM Trend Badge only
        hr_hdr = tk.Frame(hr_card, bg=PALETTE["bg_card"])
        hr_hdr.pack(fill="x", pady=(0, 6))

        IconCanvas(hr_hdr, name="heart", size=14, color=PALETTE["accent_heart"], bg=PALETTE["bg_card"]).pack(side="left", padx=(0, 6))

        tk.Label(
            hr_hdr,
            text="Heart Rate Analysis",
            font=FONTS["heading"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_primary"],
        ).pack(side="left")

        # Pill badge on top right: BPM Trend only
        tab_box = tk.Frame(hr_hdr, bg=PALETTE["bg_card"])
        tab_box.pack(side="right")

        tk.Label(
            tab_box,
            text="BPM Trend",
            font=FONTS["caption_bold"],
            bg=PALETTE["accent_blue"],
            fg=PALETTE["text_on_accent"],
            padx=10,
            pady=3,
        ).pack(side="left")

        # HR Card Body: Split into Left Hero and Right Trend Graph
        hr_body = tk.Frame(hr_card, bg=PALETTE["bg_card"])
        hr_body.pack(fill="both", expand=True)
        hr_body.columnconfigure(0, weight=4)
        hr_body.columnconfigure(1, weight=6)
        hr_body.rowconfigure(0, weight=1)

        # Left: HeartRateHeroWidget
        self.hero_vitals = HeartRateHeroWidget(hr_body)
        self.hero_vitals.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Right: BPM History Canvas
        right_graph_box = tk.Frame(hr_body, bg=PALETTE["bg_card"])
        right_graph_box.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            right_graph_box,
            text="BPM History (last 60s)",
            font=FONTS["caption_bold"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_secondary"],
        ).pack(anchor="w", pady=(0, 4))

        self.comparison_canvas = BpmComparisonCanvas(right_graph_box, width=300, height=220)
        self.comparison_canvas.pack(fill="both", expand=True)

        # Tier 2 (Middle Row): Filtered Pulse Signal (Left) + Welch PSD (Right)
        mid_row = tk.Frame(parent, bg=PALETTE["bg_root"])
        mid_row.pack(fill="both", expand=True, pady=(0, 8))
        mid_row.columnconfigure(0, weight=1)
        mid_row.columnconfigure(1, weight=1)
        mid_row.rowconfigure(0, weight=1)

        # Left: Filtered Pulse Signal
        wave_card = tk.Frame(
            mid_row,
            bg=PALETTE["bg_card"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            padx=12,
            pady=10,
        )
        wave_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        w_hdr = tk.Frame(wave_card, bg=PALETTE["bg_card"])
        w_hdr.pack(fill="x", pady=(0, 4))

        IconCanvas(w_hdr, name="pulse", size=14, color=PALETTE["accent_cyan"], bg=PALETTE["bg_card"]).pack(side="left", padx=(0, 6))

        tk.Label(
            w_hdr,
            text="Filtered Pulse Signal",
            font=FONTS["heading"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_primary"],
        ).pack(side="left")

        self.waveform_canvas = WaveformCanvas(wave_card, height=140)
        self.waveform_canvas.pack(fill="both", expand=True)

        # Right: Welch Power Spectral Density
        spec_card = tk.Frame(
            mid_row,
            bg=PALETTE["bg_card"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            padx=12,
            pady=10,
        )
        spec_card.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        s_hdr = tk.Frame(spec_card, bg=PALETTE["bg_card"])
        s_hdr.pack(fill="x", pady=(0, 4))

        IconCanvas(s_hdr, name="bars", size=14, color=PALETTE["accent_purple"], bg=PALETTE["bg_card"]).pack(side="left", padx=(0, 6))

        tk.Label(
            s_hdr,
            text="Welch Power Spectral Density (0.5 - 3.5 Hz)",
            font=FONTS["heading"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_primary"],
        ).pack(side="left")

        self.spectrum_canvas = SpectrumCanvas(spec_card, height=140)
        self.spectrum_canvas.pack(fill="both", expand=True)

        # Tier 3 (Bottom Row): 5 Metric Cards in a row
        bottom_metrics_row = tk.Frame(parent, bg=PALETTE["bg_root"])
        bottom_metrics_row.pack(fill="x")
        for col_idx in range(5):
            bottom_metrics_row.columnconfigure(col_idx, weight=1)

        # 1. Signal Quality (SQI) — first, so capture confidence is visible
        # before interpreting any heart-rate estimate.
        self.card_sqi = MetricCard(
            bottom_metrics_row,
            title="Signal Quality (SQI)",
            value="--",
            icon_name="signal_bars",
            show_bar=True,
            accent_color=PALETTE["accent_emerald"],
        )
        self.card_sqi.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # 2. Spectral FFT
        self.card_fft = MetricCard(
            bottom_metrics_row,
            title="Spectral FFT",
            value="--",
            unit="BPM",
            icon_name="bars",
            accent_color=PALETTE["accent_purple"],
        )
        self.card_fft.grid(row=0, column=1, sticky="nsew", padx=(4, 4))

        # 3. Peak Interval
        self.card_peaks = MetricCard(
            bottom_metrics_row,
            title="Peak Interval",
            value="--",
            unit="BPM",
            icon_name="pulse",
            accent_color=PALETTE["accent_cyan"],
        )
        self.card_peaks.grid(row=0, column=2, sticky="nsew", padx=(4, 4))

        # 4. Estimated Heart Rate
        self.card_bpm = MetricCard(
            bottom_metrics_row,
            title="Estimated Heart Rate",
            value="--",
            unit="BPM",
            icon_name="heart",
            accent_color=PALETTE["accent_heart"],
        )
        self.card_bpm.grid(row=0, column=3, sticky="nsew", padx=(4, 4))

        # 5. Ground Truth
        self.card_ground_truth = MetricCard(
            bottom_metrics_row,
            title="Ground Truth",
            value="--",
            unit="BPM",
            icon_name="target",
            accent_color=PALETTE["accent_purple"],
        )
        self.card_ground_truth.grid(row=0, column=4, sticky="nsew", padx=(4, 0))

    # -------------------------------------------------------------------------
    # App Logic & Engine Control
    # -------------------------------------------------------------------------
    def _toggle_monitor(self) -> None:
        """Start or stop the background monitoring engine."""
        if self.engine.is_running:
            self.engine.stop()
            self._on_monitor_stopped()
        else:
            self._start_monitor()

    def _start_monitor(self) -> None:
        mode = self.var_source_mode.get()
        if mode == "camera":
            source = self.var_camera_idx.get().strip() or "0"
        elif mode == "video":
            source = self.var_video_path.get().strip()
            if not source or not Path(source).is_file():
                messagebox.showerror("Invalid Source", "Please select a valid video file.")
                return
        elif mode == "ubfc":
            source = self.var_ubfc_path.get().strip()
            if not source or not Path(source).is_dir():
                messagebox.showerror("Invalid Source", "Please select a valid UBFC dataset directory.")
                return
        else:
            source = "0"

        method_raw = self.var_extraction_method.get().strip()
        method_map = {
            "Select All (Auto-Best SQI)": "all",
            "CHROM (Chrominance)": "chrom",
            "POS (Plane-Orthogonal-to-Skin)": "pos",
            "GREEN (Green Channel)": "green",
            "PCA (Principal Components)": "pca",
            "ICA (Independent Components)": "ica",
        }
        chosen_method = method_map.get(method_raw, "all")

        try:
            cfg = with_overrides(
                default_config(),
                source=source,
                roi_mode=self.var_roi_mode.get(),
                skin_mask_strategy=self.var_skin_strategy.get(),
                extraction_method=chosen_method,
                # Keep helpful tracking overlays enabled without exposing an
                # extra sidebar option.
                debug_roi=True,
                headless=False,
            )
        except Exception as exc:
            messagebox.showerror("Configuration Error", str(exc))
            return

        self.btn_toggle_monitor.config(text="■ Stop Monitoring", style="Stop.TButton")
        self.badge_status.set_status("MONITORING", PALETTE["accent_emerald"])
        self.lbl_status_dot.config(fg=PALETTE["accent_emerald"])
        self.lbl_status_left.config(text="Monitoring...")

        # Reset displays
        self.card_bpm.set_value("--", "Initializing sensor...")
        self.card_fft.set_value("--")
        self.card_peaks.set_value("--")
        self.card_sqi.set_value("--")
        self.card_sqi.set_notes([])
        self.card_ground_truth.set_value("--")
        self.hero_vitals.clear()
        self.waveform_canvas.clear()
        self.spectrum_canvas.clear()
        self.comparison_canvas.clear()

        self.engine.start(cfg)

    def _on_monitor_stopped(self) -> None:
        self.btn_toggle_monitor.config(text="▶ Start Monitoring", style="Start.TButton")
        self.badge_status.set_status("READY", PALETTE["text_muted"])
        self.lbl_status_dot.config(fg=PALETTE["text_muted"])
        self.lbl_status_left.config(text="Ready")
        self.lbl_face_status.config(text="● Inactive", fg=PALETTE["text_muted"])
        self.lbl_video_face_status.config(text="● Video stopped", fg=PALETTE["text_muted"])

    def _reset_to_defaults(self) -> None:
        """Reset all options and metrics back to initial default state."""
        if self.engine.is_running:
            self.engine.stop()
            self._on_monitor_stopped()

        self.var_source_mode.set("camera")
        self.var_camera_idx.set("0")
        self.var_video_path.set("")
        self.var_ubfc_path.set("")
        self.var_extraction_method.set("Select All (Auto-Best SQI)")
        self.var_roi_mode.set("multi")
        self.var_skin_strategy.set("fixed")
        self._last_telemetry = None
        self._last_frame = None
        self._update_source_details_ui()

        # Reset metrics
        self.card_bpm.set_value("--", "Waiting for signal...", PALETTE["text_primary"])
        self.card_fft.set_value("--", "Spectral peak")
        self.card_peaks.set_value("--", "Systolic peaks")
        self.card_sqi.set_value("--")
        self.card_sqi.set_notes([])
        self.card_ground_truth.set_value("--")
        self.hero_vitals.clear()

        # Reset stats
        self.lbl_face_status.config(text="● Inactive", fg=PALETTE["text_muted"])
        self.lbl_video_face_status.config(text="● Searching for face...", fg=PALETTE["accent_amber"])
        self.lbl_fps_val.config(text="--")
        self.lbl_frames_val.config(text="--")
        self.lbl_acc_frames_val.config(text="--")

        # Reset visual canvases
        self.waveform_canvas.clear()
        self.spectrum_canvas.clear()
        self.video_canvas.clear()
        self.comparison_canvas.clear()

        # Reset status displays
        self.badge_status.set_status("READY", PALETTE["text_muted"])
        self.lbl_status_dot.config(fg=PALETTE["accent_emerald"])
        self.lbl_status_left.config(text="Ready")
        self.lbl_status_right.config(text="⛶ FPS: 0.0   Frames: 0   Accepted: 0")
        self.btn_toggle_monitor.config(text="▶ Start Monitoring", style="Start.TButton")

    def _poll_engine(self) -> None:
        """Poll queued video frames and telemetry updates from background engine."""
        # 1. Update Video Frame
        frame = self.engine.get_frame()
        if frame is not None:
            self._last_frame = frame
            self.video_canvas.update_frame(frame)

        # 2. Process all incoming telemetry events
        events = self.engine.get_events()
        for evt in events:
            evt_type = evt.get("type")
            if evt_type == "status":
                text = evt.get("text", "")
                self.badge_status.set_status(text[:18].upper())
                self.lbl_status_left.config(text=text)
            elif evt_type == "telemetry":
                telemetry: GuiTelemetry = evt["data"]
                self._apply_telemetry(telemetry)
            elif evt_type == "ubfc_summary":
                report = evt.get("report", {})
                mae = report.get("mean_absolute_bpm_error")
                mae_str = f"{mae:.2f} BPM" if mae is not None else "N/A"
                achieved = report.get("mean_estimated_bpm")
                achieved_str = f"{achieved:.1f}" if achieved is not None else "N/A"
                given = report.get("mean_ground_truth_bpm")
                given_str = f"{given:.1f}" if given is not None else "N/A"
                self.card_ground_truth.set_value(
                    given_str,
                    hint=f"Final MAE: {mae_str}",
                    color=PALETTE["accent_purple"],
                )
                self.badge_status.set_status("COMPLETED", PALETTE["accent_emerald"])
                self.lbl_status_left.config(
                    text=f"Replay complete — Achieved: {achieved_str} BPM | Ground Truth: {given_str} BPM | Final MAE: {mae_str} (Saved to session output)"
                )
            elif evt_type == "finished":
                self._on_monitor_stopped()
            elif evt_type == "error":
                err = evt.get("error", "Unknown error")
                messagebox.showerror("Acquisition Error", err)
                self._on_monitor_stopped()

        # Reschedule next frame poll
        self.after(30, self._poll_engine)

    def _apply_telemetry(self, t: GuiTelemetry) -> None:
        """Update metrics and graphs from telemetry."""
        self._last_telemetry = t
        # Status
        self.badge_status.set_status("MONITORING", PALETTE["accent_emerald"])
        self.lbl_status_dot.config(fg=PALETTE["accent_emerald"])
        self.lbl_status_left.config(text="Monitoring...")

        # Statusbar & Sidebar stats
        self.lbl_status_right.config(
            text=f"⛶ FPS: {t.effective_fps:.1f}   Frames: {t.total_frames}   Accepted: {t.accepted_samples}"
        )
        self.lbl_fps_val.config(text=f"{t.effective_fps:.1f}")
        self.lbl_frames_val.config(text=str(t.total_frames))
        self.lbl_acc_frames_val.config(text=str(t.accepted_samples))

        # Face tracking status
        if t.face_detected:
            self.lbl_face_status.config(text="● Active", fg=PALETTE["accent_emerald"])
            self.lbl_video_face_status.config(text="● Face detected & locked", fg=PALETTE["accent_emerald"])
        else:
            self.lbl_face_status.config(text="● Searching...", fg=PALETTE["accent_amber"])
            self.lbl_video_face_status.config(text="● Searching for face...", fg=PALETTE["accent_amber"])

        # Top-Right Hero Vitals
        self.hero_vitals.update_vitals(t.smoothed_bpm)

        # Bottom Estimated Heart Rate Card
        if np.isfinite(t.smoothed_bpm):
            bpm_val = f"{t.smoothed_bpm:.1f}"
            self.card_bpm.set_value(bpm_val, hint="Cardiac rate tracked")
        elif "Warming" in t.status_text or "Buffering" in t.status_text:
            self.card_bpm.set_value("...", hint="Buffering window...")
        else:
            self.card_bpm.set_value("--", hint="Waiting for signal...")

        # Bottom Spectral FFT Card
        if np.isfinite(t.bpm_fft):
            self.card_fft.set_value(f"{t.bpm_fft:.1f}", hint="Spectral peak")

        # Bottom Peak Interval Card
        if np.isfinite(t.bpm_peaks):
            self.card_peaks.set_value(f"{t.bpm_peaks:.1f}", hint="Systolic peaks")

        # Bottom Signal Quality Card
        if np.isfinite(t.quality):
            q_val = f"{t.quality:.2f}"
            method_badge = f" • {t.active_method}" if t.active_method else ""
            if t.quality >= 0.8:
                hint = f"Excellent{method_badge}"
                col = PALETTE["accent_emerald"]
                notes = ["Stable facial signal", "Low motion interference"]
            elif t.quality >= 0.5:
                hint = f"Good{method_badge}"
                col = PALETTE["accent_sky"]
                notes = ["Facial signal acquired", "Minor motion detected"]
            else:
                hint = f"Fair / Motion{method_badge}"
                col = PALETTE["accent_amber"]
                notes = ["Weak pulsatile signal", "Elevated motion detected"]
            self.card_sqi.set_value(q_val, hint=hint, color=col)
            self.card_sqi.set_notes(notes)

        # Bottom Ground Truth Card
        if t.ground_truth_bpm is not None and np.isfinite(t.ground_truth_bpm):
            gt_str = f"{t.ground_truth_bpm:.1f}"
            if np.isfinite(t.smoothed_bpm):
                diff = abs(t.smoothed_bpm - t.ground_truth_bpm)
                hint = f"Δ {diff:.1f} BPM"
            else:
                hint = "UBFC reference"
            self.card_ground_truth.set_value(gt_str, hint=hint, color=PALETTE["accent_purple"])

        # Ground Truth & Heart Rate Comparison Canvas
        if t.bpm_history_times and t.bpm_history_values:
            curr_t = t.bpm_history_times[-1]
            self.comparison_canvas.update_data(
                times=t.bpm_history_times,
                estimates=t.bpm_history_values,
                current_time=curr_t,
                ground_truths=t.gt_history_values,
                has_ground_truth=t.has_ground_truth,
            )

        # Real-time Waveform Canvas
        if t.filtered_signal is not None and len(t.filtered_signal) > 0:
            self.waveform_canvas.update_signal(t.filtered_signal, t.peaks)

        # Frequency Spectrum Canvas
        if t.freqs is not None and t.power is not None:
            self.spectrum_canvas.update_spectrum(t.freqs, t.power, t.dom_freq)

    def _open_session_dir(self) -> None:
        """Open the active or latest session directory in file explorer."""
        session_dir = self.engine.session_dir
        if session_dir is None or not session_dir.exists():
            outputs_root = Path("outputs").resolve()
            if outputs_root.exists():
                candidates = sorted(outputs_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                session_dir = candidates[0] if candidates else outputs_root
            else:
                session_dir = Path(".").resolve()

        if platform.system().lower() == "windows":
            os.startfile(str(session_dir))
        elif platform.system().lower() == "darwin":
            subprocess.run(["open", str(session_dir)], check=False)
        else:
            subprocess.run(["xdg-open", str(session_dir)], check=False)

    def _on_close(self) -> None:
        """Clean shutdown when user closes window."""
        if self.engine.is_running:
            self.engine.stop()
        self.destroy()


def launch_gui() -> int:
    """Launch the rPPG monitor GUI application."""
    app = RppgAppWindow()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_gui())
