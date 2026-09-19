import numpy as np

from rppg.results import AnalysisResult
from rppg.session import CsvWriters, categorize_rejection_reason


def test_analysis_events_csv_records_accepted_and_rejected_windows(tmp_path):
    result = AnalysisResult(
        name="POS",
        quality=3.0,
        bpm_fft=72.0,
        bpm_peaks=73.0,
        fused_bpm=72.3,
        filtered=np.array([0.0, 1.0]),
        peaks=np.array([1]),
        freqs=np.array([1.2]),
        power=np.array([1.0]),
        uniform_times=np.array([0.0, 0.1]),
        fps=10.0,
        band=(0.9, 2.0),
    )
    writers = CsvWriters(tmp_path)
    writers.write_analysis_event(1.0, 4, "accepted", sample_count=100, effective_fps=29.8, result=result)
    writers.write_analysis_event(2.0, 5, "rejected", rejection_reason="low_quality", sample_count=101)
    writers.close()

    header, accepted, rejected = (tmp_path / "csv" / "analysis_events.csv").read_text(encoding="utf-8").splitlines()
    assert "rejection_reason" in header
    assert "rejection_category" in header
    assert "effective_fps" in header
    assert ",accepted," in accepted
    assert ",POS," in accepted
    assert ",rejected,signal_quality,low_quality," in rejected


def test_analysis_rejection_categories_are_stable():
    assert categorize_rejection_reason("Sample timestamp gap 1.2s exceeds limit", "rejected") == "timing_gap"
    assert categorize_rejection_reason("Sample coverage 20% is below required", "rejected") == "coverage"
    assert categorize_rejection_reason("Illumination instability exceeds limit", "rejected") == "illumination"
    assert categorize_rejection_reason("Rejected low-quality estimates", "rejected") == "signal_quality"
