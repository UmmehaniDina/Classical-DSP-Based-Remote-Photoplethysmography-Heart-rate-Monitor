"""Typed analysis result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class AnalysisResult:
    """Accepted result for one analysis window.

    Attributes use seconds for time, hertz for frequency bands, and beats per
    minute for BPM estimates. Array fields are NumPy arrays aligned to the
    current uniform analysis window unless their names describe a spectrum.
    """

    name: str
    quality: float
    bpm_fft: float
    bpm_peaks: float
    fused_bpm: float
    filtered: FloatArray
    peaks: IntArray
    freqs: FloatArray
    power: FloatArray
    uniform_times: FloatArray
    fps: float
    band: tuple[float, float]
    illumination_instability: float = float("nan")
    candidate_metadata: dict[str, Any] = field(default_factory=dict)
    quality_features: dict[str, float | int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the legacy dict-shaped representation."""
        return {
            "name": self.name,
            "quality": self.quality,
            "bpm_fft": self.bpm_fft,
            "bpm_peaks": self.bpm_peaks,
            "fused_bpm": self.fused_bpm,
            "filtered": self.filtered,
            "peaks": self.peaks,
            "freqs": self.freqs,
            "power": self.power,
            "uniform_times": self.uniform_times,
            "fps": self.fps,
            "band": self.band,
            "illumination_instability": self.illumination_instability,
            "candidate_metadata": dict(self.candidate_metadata),
            "quality_features": dict(self.quality_features),
        }
