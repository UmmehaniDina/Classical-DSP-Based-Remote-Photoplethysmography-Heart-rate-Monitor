from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

doc = Document(Path(r"C:\Users\dumme\Downloads\EEE-xxx-project-report-template (1).docx"))
body = doc._element.body
for i, child in enumerate(body):
    tag = child.tag.rsplit('}', 1)[-1]
    if tag == 'p':
        texts = [t.text or '' for t in child.iter(qn('w:t'))]
        print(i, tag, ''.join(texts)[:120])
    elif tag == 'tbl':
        print(i, tag, 'table')
    else:
        print(i, tag)
