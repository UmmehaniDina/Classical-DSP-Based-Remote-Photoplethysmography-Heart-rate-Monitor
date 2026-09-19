import csv
import json
import statistics
from pathlib import Path


for session in sorted(Path("outputs").glob("20*")):
    estimate_path = session / "csv" / "bpm_estimates.csv"
    if not estimate_path.exists():
        continue
    with estimate_path.open(encoding="utf-8", newline="") as handle:
        estimates = list(csv.DictReader(handle))
    values = []
    for row in estimates:
        try:
            values.append(float(row["smoothed_bpm"]))
        except (KeyError, TypeError, ValueError):
            pass
    payload = {
        "session": session.name,
        "bpm_count": len(values),
        "mean_bpm": round(statistics.mean(values), 3) if values else None,
        "min_bpm": round(min(values), 3) if values else None,
        "max_bpm": round(max(values), 3) if values else None,
    }
    report = session / "metadata" / "ubfc_evaluation.json"
    if report.exists():
        payload["ubfc"] = json.loads(report.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
