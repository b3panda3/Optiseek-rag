"""Extract text from Word docs. Includes paragraphs + table cells."""
import logging
from pathlib import Path
log = logging.getLogger("optiseek.rag.parsers.docx")

def parse_docx(path, corpus_root):
    from docx import Document
    rel_path = str(path.relative_to(corpus_root))
    try:
        doc = Document(str(path))
    except Exception as e:
        log.warning("DOCX open failed: %s: %s", rel_path, e); return None
    parts = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t: parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells: parts.append("\t".join(cells))
    full_text = "\n".join(parts)
    if not full_text.strip(): return None
    return [{"file": rel_path, "text": full_text, "metadata": {"type":"docx","paragraphs":len(doc.paragraphs),"tables":len(doc.tables)}}]
