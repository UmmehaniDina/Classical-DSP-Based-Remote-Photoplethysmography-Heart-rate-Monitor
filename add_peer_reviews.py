from pathlib import Path

from docx import Document


REPORT = Path(r"C:\Users\dumme\Downloads\DSP Project\Section_6_Reflection_on_Individual_and_Team_Work.docx")

reviewer = {
    "Ummehani Dina": "Md. Montashim Billaha Chisty",
    "Md. Montashim Billaha Chisty": "Ummehani Dina",
    "Mahir Nurain Shawchchow": "Israt Ferdous Sabrin",
    "Israt Ferdous Sabrin": "Mahir Nurain Shawchchow",
}

doc = Document(REPORT)
table = doc.tables[0]
for row in table.rows[1:]:
    owner = row.cells[2].text.strip()
    if owner in reviewer:
        role = row.cells[3].text.strip()
        row.cells[3].text = f"{role}; peer review by {reviewer[owner]}"

doc.save(REPORT)
print(REPORT)
