"""Extract text from PDFs using PyMuPDF. Detects WITHDRAWN/SUPERSEDED."""
import logging
from pathlib import Path
log = logging.getLogger("optiseek.rag.parsers.pdf")

def parse_pdf(path, corpus_root):
    import fitz
    rel_path = str(path.relative_to(corpus_root))
    try:
        doc = fitz.open(path)
    except Exception as e:
        if "encrypt" in str(e).lower() or "password" in str(e).lower():
            log.info("Encrypted PDF (skipping): %s", rel_path)
        else:
            log.warning("PDF open failed: %s: %s", rel_path, e)
        return None
    if doc.is_encrypted:
        log.info("Encrypted PDF (skipping): %s", rel_path); doc.close(); return None
    full_text = ""
    num_pages = len(doc)
    for page_num in range(num_pages):
        try:
            full_text += doc[page_num].get_text("text") + "\n"
        except Exception as e:
            log.warning("Page %d extraction failed for %s: %s", page_num+1, rel_path, e)
    doc.close()
    if not full_text.strip(): return None
    is_withdrawn = any(m in full_text.upper() for m in ["WITHDRAWN","SUPERSEDED","DEPRECATED"])
    return [{"file": rel_path, "text": full_text.strip(), "metadata": {"type":"pdf","pages":num_pages,"withdrawn":is_withdrawn}}]
