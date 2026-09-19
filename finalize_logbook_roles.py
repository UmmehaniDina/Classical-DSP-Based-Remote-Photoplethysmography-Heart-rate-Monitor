from pathlib import Path

from docx import Document


REPORT = Path(r"C:\Users\dumme\Downloads\DSP Project\Section_6_Reflection_on_Individual_and_Team_Work.docx")

reviewer = {
    "Ummehani Dina": "Md. Montashim Billaha Chisty",
    "Md. Montashim Billaha Chisty": "Ummehani Dina",
    "Mahir Nurain Shawchchow": "Israt Ferdous Sabrin",
    "Israt Ferdous Sabrin": "Mahir Nurain Shawchchow",
}

roles = {
    "Configured source selection": "Configuration review and integration",
    "Implemented and validated capture timestamps": "Timing and algorithm review",
    "Prepared landmark-guided": "ROI and extraction review",
    "Implemented YCrCb": "Extraction review and testing",
    "Implemented percentile-clipped": "RGB extraction review",
    "Prepared accepted-sample": "Session logging review",
    "Implemented uniform resampling": "DSP preprocessing review",
    "Implemented and compared Green": "Candidate-method review",
    "Implemented local CHROM": "Candidate-method review and integration",
    "Implemented physiological-band": "Filter-response review",
    "Implemented Welch": "Spectral-analysis review",
    "Implemented peak detection": "Peak-analysis review",
    "Integrated candidate selection": "Quality-control review",
    "Integrated FFT-peak": "Estimate-tracking review",
    "Integrated command-line": "Runner workflow review",
    "Integrated the background": "Interactive application review",
}

doc = Document(REPORT)
table = doc.tables[0]
for row in table.rows[1:]:
    owner = row.cells[2].text.strip()
    if owner not in reviewer:
        continue
    milestone = row.cells[1].text
    for prefix, role in roles.items():
        if milestone.startswith(prefix):
            row.cells[3].text = role
            break
    comment = row.cells[4].text
    if " Peer review was exchanged with " in comment:
        comment = comment.split(" Peer review was exchanged with ", 1)[0]
    if " Reviewed by " in comment:
        comment = comment.split(" Reviewed by ", 1)[0]
    row.cells[4].text = f"{comment.rstrip()} Reviewed by {reviewer[owner]}."

doc.save(REPORT)
print(REPORT)
