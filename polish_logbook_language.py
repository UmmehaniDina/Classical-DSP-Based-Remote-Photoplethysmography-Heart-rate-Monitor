from pathlib import Path

from docx import Document


REPORT = Path(r"C:\Users\dumme\Downloads\DSP Project\Section_6_Reflection_on_Individual_and_Team_Work.docx")

reviewer = {
    "Ummehani Dina": "Md. Montashim Billaha Chisty",
    "Md. Montashim Billaha Chisty": "Ummehani Dina",
    "Mahir Nurain Shawchchow": "Israt Ferdous Sabrin",
    "Israt Ferdous Sabrin": "Mahir Nurain Shawchchow",
}

milestones = {
    "Reviewed configuration structure": "Configured source selection, analysis-window duration, frequency limits, ROI settings, and output paths.",
    "Reviewed capture timestamps": "Implemented and validated capture timestamps, monotonic fallback, duplicate handling, effective FPS, and unsafe-gap rejection.",
    "Reviewed face landmarks": "Prepared landmark-guided forehead and cheek regions with coordinate smoothing and normalized motion checks.",
    "Reviewed YCrCb": "Implemented YCrCb skin masking, minimum valid-pixel thresholds, and ROI validity metadata.",
    "Reviewed percentile-clipped": "Implemented percentile-clipped RGB pooling and fixed, adaptive-chroma, and temporal-adaptive masking alternatives.",
    "Reviewed accepted-sample": "Prepared accepted-sample storage, frame events, coverage checks, and rolling-window requirements.",
    "Reviewed uniform resampling": "Implemented uniform resampling, smoothness-priors detrending, FPS-aware lambda scaling, and channel normalization.",
    "Reviewed Green": "Implemented and compared Green and PCA baseline candidates with their candidate metadata.",
    "Reviewed local CHROM": "Implemented local CHROM and POS projections with overlapping windows and local channel normalization.",
    "Reviewed band clamping": "Implemented physiological-band clamping and third-order Butterworth forward-reverse filtering.",
    "Reviewed Welch": "Implemented Welch PSD settings, Hann tapering, overlap, spectral-peak refinement, and harmonic adjustment.",
    "Reviewed peak detection": "Implemented peak detection, polarity handling, median interval estimation, and robust outlier screening.",
    "Reviewed candidate selection": "Integrated candidate selection and multi-feature quality scoring using spectral, coverage, motion, and illumination evidence.",
    "Reviewed FFT-peak": "Integrated FFT-peak fusion, jump protection, EMA/Kalman tracking options, and low-quality rejection behaviour.",
    "Reviewed session folders": "Prepared session folders, metadata snapshots, RGB and BPM CSV records, frame events, and analysis events.",
    "Reviewed UBFC": "Prepared UBFC-style reference loading, overlap-only interpolation, paired comparisons, MAE, RMSE, and bias calculation.",
    "Reviewed timing, extraction": "Validated controlled tests for timing, extraction, DSP, tracking, sessions, evaluation, and GUI behaviour.",
    "Reviewed command-line": "Integrated command-line controls, headless replay, calibration mode, and configuration-to-evidence workflow.",
    "Reviewed the background": "Integrated the background engine, bounded queues, GUI telemetry, and processing-to-interface data flow.",
    "Reviewed warm-up": "Improved warm-up, rejected, and accepted status presentation and the visibility of quality information in the application.",
    "Inspected saved": "Inspected saved output sessions and compared unpaired webcam traces with the short paired UBFC-style replay.",
    "Reviewed log completeness": "Improved log completeness, application-side evidence flow, and configuration and diagnostic behaviour.",
    "Completed the final": "Completed the final technical review of DSP settings, quality safeguards, interactive behaviour, session records, and limitations.",
}

team_roles = {
    "Ummehani Dina": "Peer validation and integration",
    "Md. Montashim Billaha Chisty": "Peer validation and integration",
    "Mahir Nurain Shawchchow": "Peer validation and integration",
    "Israt Ferdous Sabrin": "Peer validation and integration",
}

doc = Document(REPORT)
table = doc.tables[0]
for row in table.rows[1:]:
    milestone = row.cells[1].text
    for prefix, replacement in milestones.items():
        if milestone.startswith(prefix):
            row.cells[1].text = replacement
            break
    owner = row.cells[2].text.strip()
    if owner in reviewer:
        row.cells[3].text = team_roles[owner]
        comment = row.cells[4].text.rstrip()
        if "Peer review" not in comment:
            row.cells[4].text = f"{comment} Peer review was exchanged with {reviewer[owner]}."

doc.save(REPORT)
print(REPORT)
