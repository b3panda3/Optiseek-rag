"""registry.py — Dispatch to the right parser by file extension.

Each parser returns a list of "documents" — dicts with keys:
    {"file": "specs/datasheet.pdf", "text": "...", "metadata": {...}}

The corpus walker catches per-file errors and continues, so one unreadable
file does not prevent indexing the rest.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("optiseek.rag.parsers")

# Extension → parser function mapping
EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf_parser",
    ".docx": "docx_parser",
    ".xlsx": "xlsx_parser",
    ".csv": "csv_parser",
    ".txt": "text_parser",
    ".log": "text_parser",
    ".py": "text_parser",
    ".png": "image_parser",
    ".jpg": "image_parser",
    ".jpeg": "image_parser",
    ".tif": "image_parser",
    ".tiff": "image_parser",
}


def parse_file(path: Path, corpus_root: Path, vlm=None) -> list[dict] | None:
    """Parse a single file. Returns list of document dicts, or None on failure.

    NEVER raises — catches all exceptions and logs them. This is critical:
    the corpus deliberately contains unreadable files, encrypted PDFs, and
    unknown types. A walk that raises on the first bad file indexes nothing
    after it.
    """
    ext = path.suffix.lower()

    # Unknown extension — skip
    if ext not in EXTENSION_MAP:
        log.debug("Skipping unknown type: %s", path)
        return None

    # Try to read the file — catch permission errors
    try:
        # Check readability without actually opening
        if not os.access(path, os.R_OK):
            log.warning("No read permission: %s", path)
            return None
    except OSError:
        return None

    parser_name = EXTENSION_MAP[ext]

    try:
        if parser_name == "pdf_parser":
            from .pdf_parser import parse_pdf
            docs = parse_pdf(path, corpus_root)
        elif parser_name == "docx_parser":
            from .docx_parser import parse_docx
            docs = parse_docx(path, corpus_root)
        elif parser_name == "xlsx_parser":
            from .xlsx_parser import parse_xlsx
            docs = parse_xlsx(path, corpus_root)
        elif parser_name == "csv_parser":
            from .csv_parser import parse_csv
            docs = parse_csv(path, corpus_root)
        elif parser_name == "text_parser":
            from .text_parser import parse_text
            docs = parse_text(path, corpus_root)
        elif parser_name == "image_parser":
            from .image_parser import parse_image
            docs = parse_image(path, corpus_root, vlm)
        else:
            return None

        return docs
    except PermissionError:
        log.warning("Permission denied: %s", path)
        return None
    except Exception as e:
        # Catch-all: encrypted PDFs, corrupt files, unknown binary types, etc.
        log.warning("Parse failed for %s: %s: %s", path, type(e).__name__, e)
        return None


def parse_corpus(corpus_root: Path, vlm=None) -> list[dict]:
    """Walk the entire corpus tree and parse every readable file.

    Handles the four corpus traps:
    - Empty directories: skipped naturally by os.walk
    - Unknown file types: skipped (not in EXTENSION_MAP)
    - Unreadable files (chmod 000): caught by os.access + PermissionError
    - Encrypted files: caught by the parser's try/except
    """
    all_docs: list[dict] = []

    for dirpath, dirnames, filenames in os.walk(corpus_root):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            docs = parse_file(filepath, corpus_root, vlm)
            if docs:
                all_docs.extend(docs)

    return all_docs
