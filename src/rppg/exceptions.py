"""Project-specific exception hierarchy."""

from __future__ import annotations


class RppgError(Exception):
    """Base class for expected rPPG application errors."""


class ConfigurationError(RppgError):
    """Raised when CLI or configuration values are invalid."""


class WarmupError(RppgError):
    """Raised while the analysis window is still collecting usable samples."""


class LowQualitySignalError(RppgError):
    """Raised when the current signal window cannot produce a trusted BPM."""


class AcquisitionError(RppgError):
    """Raised when camera or video acquisition fails."""


class AnalysisError(RppgError):
    """Raised for expected analysis failures that are not simply warmup or quality."""
