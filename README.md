# Classical DSP Based Remote Photoplethysmography Heart Rate Monitor

This project is a software-only, classical remote photoplethysmography (rPPG) monitor developed for the BUET EEE-312 DSP-I Laboratory Project. It estimates a pulse-related heart-rate frequency from facial video using an interpretable digital signal-processing pipeline. It is intended for educational and research demonstration only; it is not a medical device and must not be used for diagnosis, treatment, or emergency decisions.

## DSP pipeline

1. Capture a live camera stream, saved video, or UBFC-style dataset recording and retain progressing timestamps.
2. Detect facial landmarks and derive forehead, left-cheek, and right-cheek regions of interest.
3. Apply YCrCb skin screening, motion checks, robust percentile-clipped RGB pooling, and accepted-sample coverage checks.
4. Resample valid RGB observations onto a uniform time grid, apply smoothness-priors detrending, and normalize channels.
5. Generate Green, PCA, CHROM, and POS pulse candidates.
6. Apply zero-phase third-order Butterworth band-pass filtering in the configured 55-200 BPM range.
7. Estimate BPM using Welch power spectral density and independent peak-interval analysis.
8. Fuse compatible estimates using spectral quality, agreement, coverage, face-detection availability, mask coverage, motion, and illumination context.
9. Present the quality-gated result in the desktop application and store session evidence for review.

## Repository layout

```text
app.py                 Desktop application launcher
configs/default.toml   Default acquisition, ROI, DSP, and quality settings
src/rppg/              Application source code
  dsp/                 Preprocessing, filtering, spectral and peak analysis
  algorithms/          Candidate construction, analysis, and tracking
  gui/                 Tkinter desktop interface and background engine
tests/                 Controlled automated test cases
```

## Requirements

- Python 3.11 or a compatible Python environment
- A webcam or a permitted saved video for interactive use
- The packages listed below

```powershell
python -m pip install numpy scipy opencv-python mediapipe pillow matplotlib pytest
```

The first run downloads the required MediaPipe face-landmarker task file into `models/face_landmarker.task` if it is not already present. The model is intentionally not stored in this repository.

## Run the desktop application

From the repository root:

```powershell
python app.py
```

Choose a live camera, saved video, or compatible UBFC-style dataset folder in the interface. Keep the face visible, use steady illumination where possible, and wait for the configured warm-up window before interpreting an accepted result.

## Run from the command line

PowerShell users can run the headless pipeline from the project root as follows:

```powershell
$env:PYTHONPATH = "src"
python -m rppg --source 0
```

Use `python -m rppg --help` to view available input, configuration, calibration, and benchmark options.

## Evaluation

The application supports UBFC-style recordings containing `vid.avi` and `ground_truth.txt`. The evaluation path aligns estimates and reference values only across their valid common time interval, then reports paired-estimate count, MAE, RMSE, and bias. Small or condition-specific runs should be treated as preliminary evidence, not as a general performance claim.

## Team

- 2206080 - Ummehani Dina
- 2206081 - Md. Montashim Billaha Chisty
- 2206082 - Mahir Nurain Shawchchow
- 2206083 - Israt Ferdous Sabrin

## Responsible use and privacy

Use only authorized recordings and datasets. Facial video and derived physiological estimates can be sensitive. Do not use the program for covert monitoring, clinical decisions, or any purpose requiring a validated medical measurement.
