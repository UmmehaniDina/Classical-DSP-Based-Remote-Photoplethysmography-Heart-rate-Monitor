"""Live plotting and OpenCV display helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .config import AnalysisConfig
from .results import AnalysisResult


def show_ubfc_final_results(report: dict[str, object], *, show_window: bool) -> None:
    """Save and optionally show the final UBFC BPM/error summary window."""
    import csv
    from pathlib import Path

    import matplotlib.pyplot as plt

    def read_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times: list[float] = []
        estimates: list[float] = []
        references: list[float] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    times.append(float(row["time_sec"]))
                    estimates.append(float(row["estimated_bpm"]))
                    references.append(float(row["ground_truth_bpm"]))
                except (KeyError, TypeError, ValueError):
                    continue
        return np.asarray(times), np.asarray(estimates), np.asarray(references)

    comparison_path = Path(str(report["comparison_csv"]))
    frame_error_path = Path(str(report["frame_error_csv"]))
    estimate_times, estimates, references = read_series(comparison_path)
    frame_times, frame_estimates, frame_references = read_series(frame_error_path)
    frame_errors = np.abs(frame_estimates - frame_references)
    graph_path = Path(str(report["final_result_graph"]))
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    try:
        fig.canvas.manager.set_window_title("UBFC Final Results")
    except Exception:
        pass
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.25))
    summary_axis = fig.add_subplot(grid[0, 0])
    bpm_axis = fig.add_subplot(grid[0, 1])
    error_axis = fig.add_subplot(grid[1, :])
    summary_axis.axis("off")

    mae = report.get("mean_absolute_bpm_error")
    final_error = "N/A" if mae is None else f"{float(mae):.2f} BPM"
    achieved = "N/A" if report.get("mean_estimated_bpm") is None else f"{float(report['mean_estimated_bpm']):.2f} BPM"
    given = "N/A" if report.get("mean_ground_truth_bpm") is None else f"{float(report['mean_ground_truth_bpm']):.2f} BPM"
    summary_axis.text(0.02, 0.92, "Final Result", fontsize=20, fontweight="bold", va="top")
    summary_axis.text(
        0.02,
        0.78,
        r"MAE = $\frac{1}{N}\sum_{i=1}^{N}|\mathrm{Achieved\ BPM}_i - \mathrm{Ground\ Truth\ BPM}_i|$",
        fontsize=12,
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E3F2FD", "edgecolor": "#1565C0"},
    )
    summary_axis.text(
        0.02,
        0.50,
        f"Achieved BPM (mean):  {achieved}\n"
        f"Ground-truth BPM (mean):  {given}\n"
        f"Final MAE:  {final_error}\n"
        f"BPM pairs used:  {report['paired_estimate_count']}",
        fontsize=13,
        va="top",
        linespacing=1.7,
    )

    if estimate_times.size:
        bpm_axis.plot(estimate_times, estimates, color="#7B1FA2", marker="o", label="Achieved BPM")
        bpm_axis.plot(estimate_times, references, color="#00897B", marker="o", label="Ground truth BPM")
        bpm_axis.legend(loc="best")
    bpm_axis.set_title("Achieved vs ground-truth BPM")
    bpm_axis.set_xlabel("Video time (s)")
    bpm_axis.set_ylabel("BPM")
    bpm_axis.grid(True, alpha=0.3)

    if frame_times.size:
        error_axis.plot(frame_times, frame_errors, color="#D84315", lw=1.1, label="Absolute error per frame")
    if mae is not None:
        error_axis.axhline(float(mae), color="#1565C0", linestyle="--", lw=2, label=f"Final MAE = {float(mae):.2f} BPM")
    error_axis.set_title("Frame-level absolute BPM error")
    error_axis.set_xlabel("Video time (s)")
    error_axis.set_ylabel("Absolute error (BPM)")
    error_axis.grid(True, alpha=0.3)
    error_axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(graph_path, dpi=160, bbox_inches="tight")
    if show_window:
        plt.show(block=True)
    else:
        plt.close(fig)


def set_axis_limits(axis: object, y_values: ArrayLike, pad: float = 0.10) -> None:
    """Set padded y-axis limits when finite values exist."""
    y_values = np.asarray(y_values)
    if len(y_values) == 0 or not np.any(np.isfinite(y_values)):
        return
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    else:
        extra = pad * (y_max - y_min)
        y_min -= extra
        y_max += extra
    axis.set_ylim(y_min, y_max)


@dataclass
class Dashboard:
    """Matplotlib dashboard wrapper. Imports pyplot only when created."""

    plt: object
    fig: object
    ax_raw: object
    ax_filt: object
    ax_spec: object
    ax_bpm: object
    line_raw_r: object
    line_raw_g: object
    line_raw_b: object
    line_filt: object
    scatter_peaks: object
    line_spec: object
    scatter_dom: object
    line_dom: object
    line_bpm: object

    @classmethod
    def create(cls, cfg: AnalysisConfig) -> "Dashboard":
        """Create and initialize the live Matplotlib dashboard."""
        import matplotlib.pyplot as plt

        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="w")
        try:
            fig.canvas.manager.set_window_title("Classical rPPG Dashboard")
        except Exception:
            pass
        ax_raw, ax_filt, ax_spec, ax_bpm = axes.flatten()

        line_raw_r, = ax_raw.plot([], [], color="#C92A2A", lw=0.8, label="R")
        line_raw_g, = ax_raw.plot([], [], color="#0D8C26", lw=1.2, label="G")
        line_raw_b, = ax_raw.plot([], [], color="#2559D9", lw=0.8, label="B")
        ax_raw.set_title("Skin-masked ROI RGB")
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("RGB intensity")
        ax_raw.set_xlim(-cfg.analysis_window_sec, 0)
        ax_raw.grid(True)

        line_filt, = ax_filt.plot([], [], color="#0D40D9", lw=1.2)
        scatter_peaks, = ax_filt.plot([], [], "rv", markersize=6)
        ax_filt.set_title("Best classical rPPG candidate")
        ax_filt.set_xlabel("Time (s)")
        ax_filt.set_ylabel("Filtered signal")
        ax_filt.set_xlim(-cfg.analysis_window_sec, 0)
        ax_filt.grid(True)

        line_spec, = ax_spec.plot([], [], color="#404040", lw=1.0)
        scatter_dom, = ax_spec.plot([], [], "ro", markersize=7)
        line_dom = ax_spec.axvline(0, color="r", linestyle="--", lw=1.1)
        ax_spec.set_title("Welch spectrum")
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("Power")
        ax_spec.set_xlim(0, 4.5)
        ax_spec.grid(True)

        line_bpm, = ax_bpm.plot([], [], color="#8C1A8C", lw=1.6)
        ax_bpm.set_title("Estimated heart rate trend")
        ax_bpm.set_xlabel("Time (s)")
        ax_bpm.set_ylabel("Estimated BPM")
        ax_bpm.set_xlim(-cfg.bpm_history_sec, 0)
        ax_bpm.set_ylim(max(30, cfg.min_accepted_bpm - 15), cfg.max_accepted_bpm + 15)
        ax_bpm.grid(True)
        plt.tight_layout()

        return cls(
            plt,
            fig,
            ax_raw,
            ax_filt,
            ax_spec,
            ax_bpm,
            line_raw_r,
            line_raw_g,
            line_raw_b,
            line_filt,
            scatter_peaks,
            line_spec,
            scatter_dom,
            line_dom,
            line_bpm,
        )

    def update(
        self,
        sample_times: ArrayLike,
        rgb_samples: ArrayLike,
        bpm_history_times: ArrayLike,
        bpm_history_values: ArrayLike,
        current_time: float,
        result: AnalysisResult,
        smoothed_bpm: float,
    ) -> None:
        """Update all dashboard plots from the latest analysis result."""
        if len(sample_times) > 0:
            times_arr = np.asarray(sample_times, dtype=np.float64)
            rgb_arr = np.asarray(rgb_samples, dtype=np.float64)
            if times_arr.size > 0 and rgb_arr.size > 0 and rgb_arr.ndim == 2 and rgb_arr.shape[1] == 3:
                rel_raw = times_arr - current_time
                self.line_raw_r.set_data(rel_raw, rgb_arr[:, 0])
                self.line_raw_g.set_data(rel_raw, rgb_arr[:, 1])
                self.line_raw_b.set_data(rel_raw, rgb_arr[:, 2])
                set_axis_limits(self.ax_raw, rgb_arr.reshape(-1))

        rel_filt = result.uniform_times - current_time
        self.line_filt.set_data(rel_filt, result.filtered)
        peaks = result.peaks
        if peaks is not None and len(peaks) > 0:
            self.scatter_peaks.set_data(rel_filt[peaks], result.filtered[peaks])
        else:
            self.scatter_peaks.set_data([], [])
        set_axis_limits(self.ax_filt, result.filtered, pad=0.15)

        freqs = result.freqs
        power = result.power
        if freqs is not None and len(freqs) > 0 and power is not None and len(power) > 0:
            self.line_spec.set_data(freqs, power)
            if np.any(np.isfinite(power)):
                self.ax_spec.set_ylim(0, max(float(np.nanmax(power)) * 1.15, 0.01))
            else:
                self.ax_spec.set_ylim(0, 1)
        else:
            self.line_spec.set_data([], [])
            self.ax_spec.set_ylim(0, 1)

        dom_freq = result.bpm_fft / 60.0
        if np.isfinite(dom_freq) and freqs is not None and len(freqs) > 0:
            dom_power = np.interp(dom_freq, freqs, power)
            self.scatter_dom.set_data([dom_freq], [dom_power])
            self.line_dom.set_xdata([dom_freq, dom_freq])
        else:
            self.scatter_dom.set_data([], [])
            self.line_dom.set_xdata([0, 0])

        if len(bpm_history_times) > 0:
            rel_hist = np.asarray(bpm_history_times, dtype=np.float64) - current_time
            self.line_bpm.set_data(rel_hist, list(bpm_history_values))
        else:
            self.line_bpm.set_data([], [])

        self.ax_bpm.set_title(f"Accepted Heart Rate: {smoothed_bpm:.1f} BPM | SQI {result.quality:.2f}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        self.plt.ioff()
        self.plt.close(self.fig)
