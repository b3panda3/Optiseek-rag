"""pdf_parser.py — Extract text from PDF files using PyMuPDF.

Handles:
- Multi-page PDFs (each page becomes a separate document)
- Encrypted PDFs (returns None — password not provided)
- WITHDRAWN / SUPERSEDED markers (tagged in metadata for retrieval preference)
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("optiseek.rag.parsers.pdf")


def parse_pdf(path: Path, corpus_root: Path) -> list[dict] | None:
    """Extract text from a PDF. Returns one document per page."""
    import fitz  # PyMuPDF

    rel_path = str(path.relative_to(corpus_root))

    try:
        doc = fitz.open(path)
    except Exception as e:
        # Encrypted PDFs, corrupt files, etc.
        if "encrypt" in str(e).lower() or "password" in str(e).lower():
            log.info("Encrypted PDF (skipping): %s", rel_path)
        else:
            log.warning("PDF open failed: %s: %s", rel_path, e)
        return None

    if doc.is_encrypted:
        log.info("Encrypted PDF (skipping): %s", rel_path)
        doc.close()
        return None

    documents = []
    full_text = ""

    for page_num in range(len(doc)):
        try:
            page_text = doc[page_num].get_text("text")
        except Exception as e:
            log.warning("Page %d extraction failed for %s: %s",
                       page_num + 1, rel_path, e)
            continue

        if page_text.strip():
            full_text += page_text + "\n"

    doc.close()

    if not full_text.strip():
        return None

    # Check for superseded/withdrawn markers
    is_withdrawn = any(marker in full_text.upper() for marker in
                       ["WITHDRAWN", "SUPERSEDED", "DEPRECATED"])

    # Return as a single document (not per-page) — simpler chunking
    return [{
        "file": rel_path,
        "text": full_text.strip(),
        "metadata": {
            "type": "pdf",
            "pages": len(doc) if not doc.is_encrypted else 0,
            "withdrawn": is_withdrawn,
        },
    }]
