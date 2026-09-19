"""Custom visual widgets, charts, and canvases for the rPPG monitor GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageTk

from .theme import FONTS, PALETTE


class PulseIconBadge(tk.Canvas):
    """Circular dark badge with neon cyan ECG pulse waveform line."""

    def __init__(self, parent: tk.Widget, size: int = 34, **kwargs) -> None:
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=PALETTE["bg_panel"],
            highlightthickness=0,
            **kwargs,
        )
        r = size // 2
        # Circle badge
        self.create_oval(2, 2, size - 2, size - 2, fill=PALETTE["badge_icon_bg"], outline=PALETTE["border"], width=1.5)
        # ECG waveform
        pts = [
            (6, r),
            (11, r),
            (14, r - 8),
            (17, r + 9),
            (21, r - 4),
            (24, r),
            (28, r),
        ]
        self.create_line(pts, fill=PALETTE["accent_cyan"], width=2.0)


class IconCanvas(tk.Canvas):
    """Clean vector icon canvas for razor-sharp clinical icons without emoji artifacts."""

    def __init__(
        self,
        parent: tk.Widget,
        name: str,
        size: int = 16,
        color: str | None = None,
        bg: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=bg or parent.cget("bg") if (bg or hasattr(parent, "cget")) else PALETTE["bg_card"],
            highlightthickness=0,
            **kwargs,
        )
        self._name = name
        self._size = size
        self._color = color or PALETTE["accent_cyan"]
        self.draw()

    def set_color(self, color: str) -> None:
        self._color = color
        self.draw()

    def draw(self) -> None:
        self.delete("all")
        s = self._size
        c = self._color

        if self._name == "camera":
            # Camera body + lens + top flash bump
            self.create_rectangle(1, 4, s - 1, s - 2, outline=c, width=1.5)
            cx, cy = s // 2, (s + 2) // 2
            r = max(int(round(s * 0.2)), 2)
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=c, width=1.4)
            self.create_line(4, 4, 6, 2, s - 6, 2, s - 4, 4, fill=c, width=1.4)
        elif self._name == "sliders":
            # Equalizer horizontal lines with slider handles
            y1 = 3
            y2 = s // 2
            y3 = s - 3
            self.create_line(1, y1, s - 1, y1, fill=c, width=1.3)
            self.create_line(1, y2, s - 1, y2, fill=c, width=1.3)
            self.create_line(1, y3, s - 1, y3, fill=c, width=1.3)
            self.create_rectangle(3, y1 - 2, 6, y1 + 2, fill=c, outline="")
            self.create_rectangle(s - 7, y2 - 2, s - 4, y2 + 2, fill=c, outline="")
            self.create_rectangle(s // 2 - 2, y3 - 2, s // 2 + 1, y3 + 2, fill=c, outline="")
        elif self._name in ("grid", "squares"):
            # 2x2 modular matrix squares
            w = (s - 4) // 2
            self.create_rectangle(1, 1, 1 + w, 1 + w, outline=c, fill="", width=1.3)
            self.create_rectangle(s - 1 - w, 1, s - 1, 1 + w, outline=c, fill="", width=1.3)
            self.create_rectangle(1, s - 1 - w, 1 + w, s - 1, outline=c, fill="", width=1.3)
            self.create_rectangle(s - 1 - w, s - 1 - w, s - 1, s - 1, outline=c, fill="", width=1.3)
        elif self._name == "target":
            # Crosshair circle target
            r = s // 2
            self.create_oval(2, 2, s - 2, s - 2, outline=c, width=1.4)
            self.create_oval(r - 1.5, r - 1.5, r + 1.5, r + 1.5, fill=c, outline="")
            self.create_line(r, 0, r, 3, fill=c, width=1.4)
            self.create_line(r, s - 3, r, s, fill=c, width=1.4)
            self.create_line(0, r, 3, r, fill=c, width=1.4)
            self.create_line(s - 3, r, s, r, fill=c, width=1.4)
        elif self._name == "heart":
            # Smooth anatomical cardiac heart shape
            cx = s // 2
            pts = [
                (cx, s - 2),
                (1, s // 2),
                (1, 4),
                (cx - 3, 2),
                (cx, 4),
                (cx + 3, 2),
                (s - 1, 4),
                (s - 1, s // 2),
            ]
            self.create_polygon(pts, fill=c, outline="", smooth=True)
        elif self._name == "pulse":
            # ECG pulse wave
            my = s // 2
            pts = [
                (1, my),
                (4, my),
                (6, my - 5),
                (9, my + 6),
                (11, my - 3),
                (13, my),
                (s - 1, my),
            ]
            self.create_line(pts, fill=c, width=1.8)
        elif self._name == "bars":
            # 3 vertical spectrum bars
            w = 2.5
            self.create_rectangle(2, s - 7, 2 + w, s - 1, fill=c, outline="")
            self.create_rectangle(6, 2, 6 + w, s - 1, fill=c, outline="")
            self.create_rectangle(11, s - 10, 11 + w, s - 1, fill=c, outline="")
        elif self._name == "signal_bars":
            # 4 signal strength staircase bars
            w = 2.0
            self.create_rectangle(1, s - 4, 1 + w, s - 1, fill=c, outline="")
            self.create_rectangle(5, s - 7, 5 + w, s - 1, fill=c, outline="")
            self.create_rectangle(9, s - 10, 9 + w, s - 1, fill=c, outline="")
            self.create_rectangle(13, 2, 13 + w, s - 1, fill=c, outline="")
        elif self._name == "folder":
            # Folder outline
            pts = [(1, 3), (6, 3), (8, 5), (s - 1, 5), (s - 1, s - 2), (1, s - 2)]
            self.create_polygon(pts, outline=c, fill="", width=1.4)
        elif self._name == "expand":
            # 4 corner brackets
            d = 4
            self.create_line(1, d, 1, 1, d, 1, fill=c, width=1.4)
            self.create_line(s - 1 - d, 1, s - 1, 1, s - 1, d, fill=c, width=1.4)
            self.create_line(1, s - 1 - d, 1, s - 1, d, s - 1, fill=c, width=1.4)
            self.create_line(s - 1 - d, s - 1, s - 1, s - 1, s - 1, s - 1 - d, fill=c, width=1.4)
        elif self._name == "refresh":
            # Circular refresh arrow
            r = s // 2
            self.create_arc(2, 2, s - 2, s - 2, start=45, extent=270, style="arc", outline=c, width=1.5)
            self.create_polygon([(s - 4, 3), (s - 1, 7), (s - 6, 7)], fill=c, outline="")
        elif self._name == "play":
            # Play triangle
            self.create_polygon([(4, 2), (s - 3, s // 2), (4, s - 2)], fill=c, outline="")


class SourceRadioButton(tk.Frame):
    """A theme-aware source selector with an unambiguous selected state."""

    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        value: str,
        variable: tk.StringVar,
        command: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=PALETTE["bg_panel"], cursor="hand2", **kwargs)
        self._value = value
        self._variable = variable
        self._command = command
        self._trace_id = variable.trace_add("write", self._sync_state)

        self.indicator = tk.Canvas(
            self,
            width=18,
            height=18,
            bg=PALETTE["bg_panel"],
            highlightthickness=0,
        )
        self.indicator.pack(side="left", padx=(0, 5))
        self.label = tk.Label(
            self,
            text=text,
            font=FONTS["body"],
            bg=PALETTE["bg_panel"],
            fg=PALETTE["text_primary"],
            anchor="w",
        )
        self.label.pack(side="left")
        for widget in (self, self.indicator, self.label):
            widget.bind("<Button-1>", self._select)
            widget.bind("<Enter>", self._hover_in)
            widget.bind("<Leave>", self._hover_out)
        self._draw()

    def _select(self, _event: tk.Event | None = None) -> None:
        if self._variable.get() != self._value:
            self._variable.set(self._value)
            if self._command is not None:
                self._command()

    def _sync_state(self, *_args: object) -> None:
        self._draw()

    def _hover_in(self, _event: tk.Event | None = None) -> None:
        self.label.config(fg=PALETTE["accent_cyan"])

    def _hover_out(self, _event: tk.Event | None = None) -> None:
        self.label.config(fg=PALETTE["text_primary"])

    def _draw(self) -> None:
        self.indicator.delete("all")
        selected = self._variable.get() == self._value
        if selected:
            self.indicator.create_oval(2, 2, 16, 16, fill=PALETTE["accent_blue"], outline="")
            self.indicator.create_oval(7, 7, 11, 11, fill=PALETTE["text_on_accent"], outline="")
        else:
            self.indicator.create_oval(3, 3, 15, 15, fill=PALETTE["bg_panel"], outline=PALETTE["text_secondary"], width=1.4)

    def destroy(self) -> None:
        self._variable.trace_remove("write", self._trace_id)
        super().destroy()


class StatusBadge(tk.Canvas):
    """Compact, reliably positioned operational state badge."""

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(
            parent,
            height=30,
            width=116,
            bg=PALETTE["bg_panel"],
            highlightthickness=0,
            **kwargs,
        )
        self._width = 116
        self._height = 30
        self._state_text = "READY"
        self._color = PALETTE["accent_emerald"]
        self.bind("<Configure>", lambda _: self.draw())
        self.draw()

    def set_status(self, text: str, color: str | None = None) -> None:
        """Update badge text and dot color."""
        self._state_text = text.upper()
        if color:
            self._color = color
        else:
            upper = self._state_text
            if "MONITOR" in upper or "READY" in upper or "DETECTED" in upper or "ACTIVE" in upper:
                self._color = PALETTE["accent_emerald"]
            elif "WARM" in upper or "WAIT" in upper or "CHECK" in upper or "BUFFER" in upper:
                self._color = PALETTE["accent_amber"]
            elif "ERROR" in upper or "LOST" in upper or "FAIL" in upper or "STOP" in upper:
                self._color = PALETTE["accent_heart"]
            else:
                self._color = PALETTE["text_muted"]
        self.draw()

    def draw(self) -> None:
        """Render the pill badge with subtle border and glowing dot."""
        self.delete("all")
        # Before the widget is packed Tk reports a 1px size; use the requested
        # dimensions until its first configure event so the initial READY pill
        # does not appear detached from the header.
        w = max(self.winfo_width(), self._width)
        h = max(self.winfo_height(), self._height)

        # Rounded pill container
        r = (h - 4) // 2
        bg_col = PALETTE["bg_card_inner"]
        border_col = self._color if "MONITOR" in self._state_text else PALETTE["border"]

        # Build the fill first, then draw outline-only arcs. A pie-slice arc
        # has radial edges that show up as an unwanted line through the label.
        self.create_oval(2, 2, 2 + 2 * r, h - 2, fill=bg_col, outline="")
        self.create_oval(w - 2 * r - 2, 2, w - 2, h - 2, fill=bg_col, outline="")
        self.create_rectangle(2 + r, 2, w - r - 2, h - 2, fill=bg_col, outline="")
        self.create_arc(2, 2, 2 + 2 * r, h - 2, start=90, extent=180, style="arc", outline=border_col)
        self.create_arc(w - 2 * r - 2, 2, w - 2, h - 2, start=270, extent=180, style="arc", outline=border_col)
        self.create_line(2 + r, 2, w - r - 2, 2, fill=border_col)
        self.create_line(2 + r, h - 2, w - r - 2, h - 2, fill=border_col)

        # Status dot
        dot_r = 3.5
        cx = 16
        cy = h // 2
        self.create_oval(cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r, fill=self._color, outline="")

        # Status text
        self.create_text(
            cx + 10,
            cy,
            anchor="w",
            text=self._state_text,
            fill=PALETTE["text_primary"] if self._color == PALETTE["accent_emerald"] else PALETTE["text_secondary"],
            font=FONTS["body_bold"],
        )


class QualityProgressBar(tk.Canvas):
    """Sleek horizontal glowing progress bar for signal quality."""

    def __init__(self, parent: tk.Widget, width: int = 150, height: int = 7, **kwargs) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=PALETTE["bg_card"],
            highlightthickness=0,
            **kwargs,
        )
        self._width = width
        self._height = height
        self._fraction = 0.0
        self._color = PALETTE["accent_emerald"]
        self.bind("<Configure>", lambda _: self.draw())
        self.draw()

    def set_fraction(self, fraction: float, color: str | None = None) -> None:
        """Set normalized quality fraction between 0.0 and 1.0."""
        self._fraction = max(0.0, min(1.0, float(fraction)))
        if color:
            self._color = color
        self.draw()

    def draw(self) -> None:
        self.delete("all")
        w = self.winfo_width() or self._width
        h = self.winfo_height() or self._height

        # Track background
        self.create_rectangle(0, 0, w, h, fill=PALETTE["progress_track"], outline="")

        # Filled bar
        fill_w = int(w * self._fraction)
        if fill_w > 0:
            self.create_rectangle(0, 0, fill_w, h, fill=self._color, outline="")


class VideoCanvas(tk.Canvas):
    """High-performance frame canvas supporting letterboxing and smooth scaling."""

    def __init__(self, parent: tk.Widget, width: int = 440, height: int = 300, **kwargs) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=PALETTE["bg_card_inner"],
            highlightthickness=0,
            **kwargs,
        )
        self._photo_image: ImageTk.PhotoImage | None = None
        self._placeholder_text = "No Camera or Video Active"
        self._show_placeholder()
        self.bind("<Configure>", lambda _: self._on_resize())

    def _show_placeholder(self) -> None:
        self.delete("all")
        w = self.winfo_width() or 440
        h = self.winfo_height() or 300
        self.create_rectangle(0, 0, w, h, fill=PALETTE["bg_card_inner"], outline="")
        self.create_text(
            w // 2,
            h // 2 - 16,
            text="📹",
            font=(FONTS["body"][0], 28),
            fill=PALETTE["text_muted"],
        )
        self.create_text(
            w // 2,
            h // 2 + 18,
            text=self._placeholder_text,
            font=FONTS["body_bold"],
            fill=PALETTE["text_secondary"],
        )
        self.create_text(
            w // 2,
            h // 2 + 38,
            text="Select input source and press 'Start Monitoring'",
            font=FONTS["caption"],
            fill=PALETTE["text_muted"],
        )

    def _on_resize(self) -> None:
        if self._photo_image is None:
            self._show_placeholder()

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Render a raw OpenCV BGR frame into the canvas, preserving the natural shape of the video."""
        canvas_w = max(self.winfo_width(), 100)
        canvas_h = max(self.winfo_height(), 100)

        fh, fw = frame_bgr.shape[:2]
        if fw == 0 or fh == 0:
            return

        scale = min(canvas_w / fw, canvas_h / fh)
        nw = max(int(round(fw * scale)), 1)
        nh = max(int(round(fh * scale)), 1)

        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        self._photo_image = ImageTk.PhotoImage(image=pil_image)

        self.delete("all")
        self.create_rectangle(0, 0, canvas_w, canvas_h, fill=PALETTE["bg_card_inner"], outline="")
        x_offset = (canvas_w - nw) // 2
        y_offset = (canvas_h - nh) // 2
        self.create_image(x_offset, y_offset, anchor="nw", image=self._photo_image)

    def clear(self) -> None:
        """Clear canvas and display placeholder."""
        self._photo_image = None
        self._show_placeholder()


class WaveformCanvas(tk.Canvas):
    """Real-time pulse signal oscilloscope canvas with cyan line and coral systolic peaks."""

    def __init__(self, parent: tk.Widget, height: int = 150, **kwargs) -> None:
        super().__init__(
            parent,
            height=height,
            bg=PALETTE["bg_card_inner"],
            highlightthickness=0,
            **kwargs,
        )
        self._height = height
        self._data: np.ndarray | None = None
        self._peaks: np.ndarray | None = None
        self.bind("<Configure>", lambda _: self.redraw())

    def update_signal(self, signal: np.ndarray, peaks: np.ndarray | None = None) -> None:
        """Update waveform data and trigger redraw."""
        self._data = np.asarray(signal, dtype=np.float64)
        self._peaks = np.asarray(peaks) if peaks is not None else None
        self.redraw()

    def clear(self) -> None:
        """Reset waveform display."""
        self._data = None
        self._peaks = None
        self.delete("all")
        self._draw_empty_grid()

    def _draw_empty_grid(self) -> None:
        w = self.winfo_width() or 400
        h = self.winfo_height() or self._height

        pad_left = 38
        pad_right = 16
        pad_top = 22
        pad_bottom = 20

        usable_w = w - pad_left - pad_right
        usable_h = h - pad_top - pad_bottom

        # Y Ticks: 1.0, 0.5, 0.0, -0.5, -1.0
        y_levels = [1.0, 0.5, 0.0, -0.5, -1.0]
        for y_val in y_levels:
            y_frac = (y_val + 1.0) / 2.0
            y = int(pad_top + usable_h - y_frac * usable_h)
            self.create_line(pad_left, y, w - pad_right, y, fill=PALETTE["border_subtle"], dash=(1, 5))
            self.create_text(
                pad_left - 6,
                y,
                anchor="e",
                text=f"{y_val:.1f}",
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        # X Ticks: 0s, 2s, 4s, 6s, 8s, 10s
        for sec in [0, 2, 4, 6, 8, 10]:
            x = int(pad_left + (sec / 10.0) * usable_w)
            self.create_text(
                x,
                h - 8,
                anchor="s",
                text=f"{sec}s",
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        self._draw_legend(w)

        self.create_text(
            (pad_left + w) // 2,
            (pad_top + h) // 2,
            text="Waiting for pulse signal...",
            fill=PALETTE["text_muted"],
            font=FONTS["caption"],
        )

    def _draw_legend(self, w: int) -> None:
        # Top right legend: — Filtered Signal  ● Systolic Peaks
        leg_y = 12
        leg_right = w - 16

        # Systolic peaks dot
        self.create_oval(leg_right - 105, leg_y - 3, leg_right - 99, leg_y + 3, fill=PALETTE["accent_heart"], outline="")
        self.create_text(
            leg_right - 94,
            leg_y,
            anchor="w",
            text="Systolic Peaks",
            fill=PALETTE["text_secondary"],
            font=FONTS["caption"],
        )

        # Filtered signal line
        self.create_line(leg_right - 210, leg_y, leg_right - 198, leg_y, fill=PALETTE["accent_cyan"], width=2)
        self.create_text(
            leg_right - 193,
            leg_y,
            anchor="w",
            text="Filtered Signal",
            fill=PALETTE["text_secondary"],
            font=FONTS["caption"],
        )

    def redraw(self) -> None:
        """Redraw the waveform line and peaks."""
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 10 or h <= 10:
            return

        if self._data is None or len(self._data) < 2:
            self._draw_empty_grid()
            return

        pad_left = 38
        pad_right = 16
        pad_top = 22
        pad_bottom = 20

        usable_w = w - pad_left - pad_right
        usable_h = h - pad_top - pad_bottom

        self._draw_legend(w)

        # Y Axis Ticks
        for y_val in [1.0, 0.5, 0.0, -0.5, -1.0]:
            y_frac = (y_val + 1.0) / 2.0
            y = int(pad_top + usable_h - y_frac * usable_h)
            dash_style = (2, 4) if y_val == 0.0 else (1, 5)
            self.create_line(pad_left, y, w - pad_right, y, fill=PALETTE["border_subtle"], dash=dash_style)
            self.create_text(
                pad_left - 6,
                y,
                anchor="e",
                text=f"{y_val:.1f}",
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        # X Axis Ticks
        for sec in [0, 2, 4, 6, 8, 10]:
            x = int(pad_left + (sec / 10.0) * usable_w)
            self.create_text(
                x,
                h - 8,
                anchor="s",
                text=f"{sec}s",
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        # Scale data dynamically to roughly [-1.0, 1.0]
        data = self._data
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            self._draw_empty_grid()
            return

        # Normalize signal around zero
        std_val = float(np.nanstd(data))
        if std_val > 1e-6:
            mean_val = float(np.nanmean(data))
            norm_data = (data - mean_val) / (2.8 * std_val)
            norm_data = np.clip(norm_data, -1.0, 1.0)
        else:
            norm_data = np.zeros_like(data)

        n_points = len(norm_data)
        coords = []
        point_positions = []
        for i, val in enumerate(norm_data):
            x = int(pad_left + (i / (n_points - 1)) * usable_w)
            y_frac = (val + 1.0) / 2.0
            y = int(pad_top + usable_h - y_frac * usable_h)
            coords.extend([x, y])
            point_positions.append((x, y))

        # Cyan pulse waveform
        self.create_line(coords, fill=PALETTE["accent_cyan"], width=2, smooth=True)

        # Draw systolic peak markers (coral red)
        if self._peaks is not None and len(self._peaks) > 0:
            for pk in self._peaks:
                if 0 <= pk < len(point_positions):
                    px, py = point_positions[pk]
                    pr = 3.5
                    self.create_oval(
                        px - pr,
                        py - pr,
                        px + pr,
                        py + pr,
                        fill=PALETTE["accent_heart"],
                        outline=PALETTE["plot_marker_outline"],
                        width=1,
                    )


class SpectrumCanvas(tk.Canvas):
    """Welch Power Spectral Density canvas with shaded purple peak curve and drop-line."""

    def __init__(self, parent: tk.Widget, height: int = 150, **kwargs) -> None:
        super().__init__(
            parent,
            height=height,
            bg=PALETTE["bg_card_inner"],
            highlightthickness=0,
            **kwargs,
        )
        self._height = height
        self._freqs: np.ndarray | None = None
        self._power: np.ndarray | None = None
        self._dom_freq: float | None = None
        self.bind("<Configure>", lambda _: self.redraw())

    def update_spectrum(
        self, freqs: np.ndarray, power: np.ndarray, dom_freq: float | None = None
    ) -> None:
        """Update frequency spectrum data."""
        self._freqs = np.asarray(freqs, dtype=np.float64)
        self._power = np.asarray(power, dtype=np.float64)
        self._dom_freq = dom_freq
        self.redraw()

    def clear(self) -> None:
        """Reset spectrum."""
        self._freqs = None
        self._power = None
        self._dom_freq = None
        self.delete("all")
        self._draw_empty_grid()

    def _draw_empty_grid(self) -> None:
        w = self.winfo_width() or 400
        h = self.winfo_height() or self._height

        pad_left = 36
        pad_right = 16
        pad_top = 22
        pad_bottom = 20

        usable_w = w - pad_left - pad_right
        usable_h = h - pad_top - pad_bottom

        # Y Ticks: 2.0, 1.5, 1.0, 0.5, 0.0
        for y_val in [2.0, 1.5, 1.0, 0.5, 0.0]:
            y_frac = y_val / 2.0
            y = int(pad_top + usable_h - y_frac * usable_h)
            self.create_line(pad_left, y, w - pad_right, y, fill=PALETTE["border_subtle"], dash=(1, 5))
            self.create_text(
                pad_left - 6,
                y,
                anchor="e",
                text=f"{y_val:.1f}",
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        # X Ticks: 0.5Hz, 1.0Hz, 1.5Hz, 2.0Hz, 2.5Hz, 3.0Hz, 3.5Hz
        f_min, f_max = 0.5, 3.5
        for hz in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            x = int(pad_left + ((hz - f_min) / (f_max - f_min)) * usable_w)
            self.create_text(
                x,
                h - 8,
                anchor="s",
                text=f"{hz:.1f}Hz",
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        self.create_text(
            (pad_left + w) // 2,
            (pad_top + h) // 2,
            text="Waiting for spectral analysis...",
            fill=PALETTE["text_muted"],
            font=FONTS["caption"],
        )

    def redraw(self) -> None:
        """Redraw spectral power curve, shaded fill, and dominant cardiac peak."""
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 10 or h <= 10:
            return

        if self._freqs is None or self._power is None or len(self._freqs) < 2:
            self._draw_empty_grid()
            return

        pad_left = 36
        pad_right = 16
        pad_top = 22
        pad_bottom = 20

        usable_w = w - pad_left - pad_right
        usable_h = h - pad_top - pad_bottom

        f_min, f_max = 0.5, 3.5
        mask = (self._freqs >= f_min) & (self._freqs <= f_max)
        if not np.any(mask):
            mask = np.ones_like(self._freqs, dtype=bool)
            f_min = float(self._freqs[0])
            f_max = float(self._freqs[-1])

        freqs = self._freqs[mask]
        power = self._power[mask]

        if len(freqs) < 2 or not np.any(np.isfinite(power)):
            self._draw_empty_grid()
            return

        p_max = float(np.nanmax(power))
        if p_max <= 0:
            p_max = 1.0

        # Y Ticks
        for y_val in [2.0, 1.5, 1.0, 0.5, 0.0]:
            y_frac = y_val / 2.0
            y = int(pad_top + usable_h - y_frac * usable_h)
            self.create_line(pad_left, y, w - pad_right, y, fill=PALETTE["border_subtle"], dash=(1, 5))
            self.create_text(
                pad_left - 6,
                y,
                anchor="e",
                text=f"{y_val:.1f}",
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        # X Ticks
        for hz in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            x = int(pad_left + ((hz - f_min) / (f_max - f_min)) * usable_w)
            self.create_text(
                x,
                h - 8,
                anchor="s",
                text=f"{hz:.1f}Hz",
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        axis_y = pad_top + usable_h
        coords = [pad_left, axis_y]
        for f, p in zip(freqs, power):
            x = int(pad_left + ((f - f_min) / (f_max - f_min)) * usable_w)
            y = int(axis_y - (p / p_max) * usable_h)
            coords.extend([x, y])
        coords.extend([pad_left + usable_w, axis_y])

        # Fill curve polygon (shaded purple)
        self.create_polygon(coords, fill=PALETTE["accent_purple_fill"], outline="", smooth=True)

        # Stroke curve line (vibrant purple)
        curve_pts = coords[2:-2]
        if len(curve_pts) >= 4:
            self.create_line(curve_pts, fill=PALETTE["accent_purple"], width=2.2, smooth=True)

        # Dominant peak line (coral red dashed drop-line) and peak label
        if self._dom_freq is not None and f_min <= self._dom_freq <= f_max:
            dom_x = int(pad_left + ((self._dom_freq - f_min) / (f_max - f_min)) * usable_w)
            self.create_line(dom_x, pad_top + 4, dom_x, axis_y, fill=PALETTE["accent_heart"], width=1.5, dash=(2, 2))
            peak_bpm = self._dom_freq * 60.0
            self.create_text(
                dom_x,
                pad_top + 2,
                anchor="s",
                text=f"Peak: {peak_bpm:.1f} BPM",
                fill=PALETTE["accent_purple_label"],
                font=FONTS["caption_bold"],
            )


class BpmComparisonCanvas(tk.Canvas):
    """Real-time Heart Rate Trend and Ground-Truth Comparison graph canvas."""

    def __init__(self, parent: tk.Widget, height: int = 240, **kwargs) -> None:
        super().__init__(
            parent,
            height=height,
            bg=PALETTE["bg_card_inner"],
            highlightthickness=0,
            **kwargs,
        )
        self._height = height
        self._times: list[float] = []
        self._estimates: list[float] = []
        self._references: list[float] = []
        self._current_time: float = 0.0
        self._has_ground_truth: bool = False
        self.bind("<Configure>", lambda _: self.redraw())

    def update_data(
        self,
        times: Sequence[float],
        estimates: Sequence[float],
        current_time: float,
        ground_truths: Sequence[float] | None = None,
        has_ground_truth: bool = False,
    ) -> None:
        """Update trend line with latest history and ground truth."""
        self._times = list(times)
        self._estimates = list(estimates)
        self._current_time = current_time
        self._has_ground_truth = has_ground_truth
        self._references = list(ground_truths) if ground_truths is not None else []
        self.redraw()

    def clear(self) -> None:
        """Reset comparison canvas."""
        self._times.clear()
        self._estimates.clear()
        self._references.clear()
        self._has_ground_truth = False
        self.delete("all")
        self._draw_empty_grid()

    def _draw_legend(self, has_reference: bool) -> None:
        """Identify the trend colors without relying on the numeric callout."""
        y = 12
        x = 38
        self.create_line(x, y, x + 15, y, fill=PALETTE["accent_cyan"], width=2.4)
        self.create_text(
            x + 20,
            y,
            anchor="w",
            text="Estimate",
            fill=PALETTE["text_secondary"],
            font=FONTS["caption"],
        )
        if has_reference:
            x = 112
            self.create_line(x, y, x + 15, y, fill=PALETTE["accent_purple"], width=2.0, dash=(4, 2))
            self.create_text(
                x + 20,
                y,
                anchor="w",
                text="Reference",
                fill=PALETTE["text_secondary"],
                font=FONTS["caption"],
            )

    def _draw_empty_grid(self) -> None:
        w = self.winfo_width() or 400
        h = self.winfo_height() or self._height

        pad_left = 36
        pad_right = 16
        pad_top = 26
        pad_bottom = 22

        usable_w = w - pad_left - pad_right
        usable_h = h - pad_top - pad_bottom

        self._draw_legend(False)

        # Y Axis Ticks: 120, 100, 80, 60, 40
        for bpm in [120, 100, 80, 60, 40]:
            y_frac = (bpm - 40) / 80.0
            y = int(pad_top + usable_h - y_frac * usable_h)
            self.create_line(pad_left, y, w - pad_right, y, fill=PALETTE["border_subtle"], dash=(1, 5))
            self.create_text(
                pad_left - 6, y, anchor="e", text=str(bpm), fill=PALETTE["text_muted"], font=FONTS["caption"]
            )

        # X Axis Ticks: -60s, -50s, -40s, -30s, -20s, -10s, Now
        for sec in [-60, -50, -40, -30, -20, -10, 0]:
            x = int(pad_left + ((sec + 60) / 60.0) * usable_w)
            lbl = "Now" if sec == 0 else f"{sec}s"
            self.create_text(
                x,
                h - 8,
                anchor="s",
                text=lbl,
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        self.create_text(
            (pad_left + w) // 2,
            (pad_top + h) // 2,
            text="Awaiting heart rate estimates...",
            fill=PALETTE["text_muted"],
            font=FONTS["caption"],
        )

    def redraw(self) -> None:
        """Redraw comparison trend line and ground truth if available."""
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 60 or h <= 60:
            return

        if not self._times or not self._estimates:
            self._draw_empty_grid()
            return

        pad_left = 36
        pad_right = 16
        pad_top = 26
        pad_bottom = 22

        usable_w = w - pad_left - pad_right
        usable_h = h - pad_top - pad_bottom

        self._draw_legend(self._has_ground_truth and bool(self._references))

        # Y scale fixed: 40 to 120 BPM (or expands if out of bounds)
        valid_est = [v for v in self._estimates if np.isfinite(v)]
        valid_gt = [v for v in self._references if np.isfinite(v)]
        all_vals = valid_est + (valid_gt if self._has_ground_truth else [])
        if not all_vals:
            self._draw_empty_grid()
            return

        y_min = 40.0
        y_max = 120.0
        if min(all_vals) < 40.0:
            y_min = float(np.floor(min(all_vals) / 10.0) * 10.0)
        if max(all_vals) > 120.0:
            y_max = float(np.ceil(max(all_vals) / 10.0) * 10.0)
        y_range = max(y_max - y_min, 20.0)

        # Y Ticks
        for bpm in range(int(y_min), int(y_max) + 1, 20):
            y_frac = (bpm - y_min) / y_range
            y = int(pad_top + usable_h - y_frac * usable_h)
            self.create_line(pad_left, y, w - pad_right, y, fill=PALETTE["border_subtle"], dash=(1, 5))
            self.create_text(
                pad_left - 6,
                y,
                anchor="e",
                text=str(bpm),
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        # X Ticks (last 60s)
        for sec in [-60, -50, -40, -30, -20, -10, 0]:
            x = int(pad_left + ((sec + 60) / 60.0) * usable_w)
            lbl = "Now" if sec == 0 else f"{sec}s"
            self.create_text(
                x,
                h - 8,
                anchor="s",
                text=lbl,
                fill=PALETTE["text_muted"],
                font=FONTS["caption"],
            )

        t_now = self._current_time
        t_span = 60.0

        # Plot Ground Truth Line (if present)
        gt_last_val = None
        if self._has_ground_truth and self._references:
            gt_coords = []
            for t_val, gt_val in zip(self._times, self._references):
                if np.isfinite(gt_val) and (t_now - t_val) <= t_span:
                    rel_t = max(0.0, min(t_span, t_span - (t_now - t_val)))
                    x = int(pad_left + (rel_t / t_span) * usable_w)
                    y = int(pad_top + usable_h - ((gt_val - y_min) / y_range) * usable_h)
                    gt_coords.extend([x, y])
                    gt_last_val = gt_val
            if len(gt_coords) >= 4:
                self.create_line(
                    gt_coords,
                    fill=PALETTE["accent_purple"],
                    width=2.0,
                    dash=(4, 2),
                    smooth=True,
                    splinesteps=12,
                )

        # Plot an interpolated trend line. BPM estimates arrive as discrete
        # samples, but the visual should communicate one continuous signal.
        est_coords = []
        est_last_val = None
        for t_val, est_val in zip(self._times, self._estimates):
            if np.isfinite(est_val) and (t_now - t_val) <= t_span:
                rel_t = max(0.0, min(t_span, t_span - (t_now - t_val)))
                x = int(pad_left + (rel_t / t_span) * usable_w)
                y = int(pad_top + usable_h - ((est_val - y_min) / y_range) * usable_h)
                est_coords.extend([x, y])
                est_last_val = est_val

        if len(est_coords) >= 4:
            self.create_line(
                est_coords,
                fill=PALETTE["accent_cyan"],
                width=2.4,
                smooth=True,
                splinesteps=16,
            )
        elif len(est_coords) == 2:
            ex, ey = est_coords[0], est_coords[1]
            self.create_line(ex - 1, ey, ex + 1, ey, fill=PALETTE["accent_cyan"], width=2.4)



class HeartRateHeroWidget(tk.Frame):
    """Sub-panel for Heart Rate Analysis displaying the current BPM only."""

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, bg=PALETTE["bg_card"], **kwargs)

        # 1. Heart icon + Title
        hdr_row = tk.Frame(self, bg=PALETTE["bg_card"])
        hdr_row.pack(fill="x", anchor="w", pady=(0, 4))

        lbl_icon = IconCanvas(
            hdr_row,
            name="heart",
            size=14,
            color=PALETTE["accent_heart"],
            bg=PALETTE["bg_card"],
        )
        lbl_icon.pack(side="left", padx=(0, 6))

        lbl_title = tk.Label(
            hdr_row,
            text="Current BPM",
            font=FONTS["caption_bold"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_secondary"],
        )
        lbl_title.pack(side="left")

        # 2. Hero BPM + Normal Badge
        val_row = tk.Frame(self, bg=PALETTE["bg_card"])
        val_row.pack(fill="x", anchor="w", pady=(0, 2))

        self.lbl_bpm_val = tk.Label(
            val_row,
            text="--",
            font=FONTS["bpm_large"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_primary"],
        )
        self.lbl_bpm_val.pack(side="left", anchor="s")

        self.lbl_bpm_unit = tk.Label(
            val_row,
            text=" BPM",
            font=FONTS["bpm_unit"],
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_secondary"],
        )
        self.lbl_bpm_unit.pack(side="left", anchor="s", padx=(0, 10), pady=(0, 6))

        # Normal badge (green pill)
        self.badge_normal = tk.Label(
            val_row,
            text="Normal",
            font=FONTS["caption_bold"],
            bg=PALETTE["accent_emerald_bg"],
            fg=PALETTE["accent_emerald"],
            padx=8,
            pady=2,
            relief="solid",
            borderwidth=1,
        )
        self.badge_normal.pack(side="left", anchor="s", pady=(0, 8))

    def update_vitals(self, bpm: float | None) -> None:
        """Update hero card readings."""
        if bpm is not None and np.isfinite(bpm) and bpm > 0:
            self.lbl_bpm_val.config(text=f"{bpm:.1f}")
            if 60.0 <= bpm <= 100.0:
                self.badge_normal.config(text="Normal", bg=PALETTE["accent_emerald_bg"], fg=PALETTE["accent_emerald"])
            elif bpm > 100.0:
                self.badge_normal.config(text="High", bg=PALETTE["accent_warning_bg"], fg=PALETTE["accent_amber"])
            else:
                self.badge_normal.config(text="Low", bg=PALETTE["accent_warning_bg"], fg=PALETTE["accent_amber"])
        else:
            self.lbl_bpm_val.config(text="--")
            self.badge_normal.config(text="--", bg=PALETTE["badge_idle_bg"], fg=PALETTE["text_muted"])

    def clear(self) -> None:
        """Reset hero panel to defaults."""
        self.lbl_bpm_val.config(text="--")
        self.badge_normal.config(text="Normal", bg=PALETTE["accent_emerald_bg"], fg=PALETTE["accent_emerald"])


class MetricCard(tk.Frame):
    """Refined dark card showing an individual health telemetry metric matching the mockup."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        value: str = "--",
        unit: str = "",
        accent_color: str | None = None,
        large: bool = False,
        icon: str = "",
        icon_name: str = "",
        show_bar: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            bg=PALETTE["bg_card"],
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            padx=14,
            pady=10,
            **kwargs,
        )
        self._accent = accent_color or PALETTE["accent_sky"]
        self._large = large
        self._show_bar = show_bar

        # Title / Label
        hdr = tk.Frame(self, bg=PALETTE["bg_card"])
        hdr.pack(fill="x", anchor="w")

        if icon_name:
            ico = IconCanvas(
                hdr,
                name=icon_name,
                size=14,
                color=self._accent if self._accent else PALETTE["accent_cyan"],
                bg=PALETTE["bg_card"],
            )
            ico.pack(side="left", padx=(0, 6))
        elif icon:
            lbl_ico = tk.Label(
                hdr,
                text=icon,
                font=(FONTS["caption"][0], 10),
                bg=PALETTE["bg_card"],
                fg=PALETTE["accent_cyan"],
            )
            lbl_ico.pack(side="left", padx=(0, 6))

        self.lbl_title = tk.Label(
            hdr,
            text=title.upper(),
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_secondary"],
            font=FONTS["caption_bold"],
            anchor="w",
        )
        self.lbl_title.pack(side="left", anchor="w")

        # Value container
        val_frame = tk.Frame(self, bg=PALETTE["bg_card"])
        val_frame.pack(fill="x", anchor="w", pady=(3, 0))

        val_font = FONTS["bpm_large"] if large else FONTS["metric_value"]
        self.lbl_value = tk.Label(
            val_frame,
            text=value,
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_primary"],
            font=val_font,
        )
        self.lbl_value.pack(side="left", anchor="s")

        if unit:
            unit_font = FONTS["bpm_unit"] if large else FONTS["caption_bold"]
            self.lbl_unit = tk.Label(
                val_frame,
                text=f" {unit}",
                bg=PALETTE["bg_card"],
                fg=PALETTE["text_muted"],
                font=unit_font,
            )
            self.lbl_unit.pack(side="left", anchor="s", padx=(2, 0), pady=(0, 2))
        else:
            self.lbl_unit = tk.Label(val_frame, text="", bg=PALETTE["bg_card"])

        # Optional bottom status/hint or progress bar
        self.lbl_hint = tk.Label(
            self,
            text="",
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_secondary"],
            font=FONTS["caption"],
            anchor="w",
        )
        self.lbl_hint.pack(fill="x", anchor="w", pady=(2, 0))

        if self._show_bar:
            self.bar = QualityProgressBar(self, width=120, height=5)
            self.bar.pack(fill="x", anchor="w", pady=(3, 0))
            self.notes_frame = tk.Frame(self, bg=PALETTE["bg_card"])
            self.note_labels = []
            for _ in range(2):
                label = tk.Label(
                    self.notes_frame,
                    bg=PALETTE["bg_card"],
                    fg=PALETTE["text_secondary"],
                    font=FONTS["caption"],
                    anchor="w",
                )
                label.pack(fill="x", anchor="w")
                self.note_labels.append(label)
        else:
            self.bar = None
            self.notes_frame = None
            self.note_labels = []

    def set_value(self, value: str, hint: str = "", color: str | None = None) -> None:
        """Update value string and optional hint."""
        self.lbl_value.config(text=value)
        if color:
            self.lbl_value.config(fg=color)
        else:
            self.lbl_value.config(fg=PALETTE["text_primary"])
        if hint:
            self.lbl_hint.config(text=hint)
            self.lbl_hint.pack(fill="x", anchor="w", pady=(2, 0))
        else:
            self.lbl_hint.config(text="")

        if self.bar is not None:
            try:
                num = float(value)
                frac = num if num <= 1.0 else num / 20.0
                self.bar.set_fraction(frac)
            except Exception:
                self.bar.set_fraction(0.0)

    def set_notes(self, notes: Sequence[str]) -> None:
        """Show up to two concise quality notes below a quality meter."""
        if self.notes_frame is None:
            return
        for index, label in enumerate(self.note_labels):
            label.config(text=(f"• {notes[index]}" if index < len(notes) else ""))
        if notes:
            self.notes_frame.pack(fill="x", anchor="w", pady=(4, 0))
        else:
            self.notes_frame.pack_forget()
