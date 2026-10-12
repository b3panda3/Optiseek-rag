"""text_parser.py — Extract text from plain text files.

Handles .txt, .log, and .py files. All are read as UTF-8 with fallback
encoding for production logs that may use latin-1.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("optiseek.rag.parsers.text")


def parse_text(path: Path, corpus_root: Path) -> list[dict] | None:
    """Read a plain text file (.txt, .log, .py)."""
    rel_path = str(path.relative_to(corpus_root))

    try:
        # Try UTF-8 first, fall back to latin-1 (never fails)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
    except Exception as e:
        log.warning("Text read failed: %s: %s", rel_path, e)
        return None

    if not text.strip():
        return None

    return [{
        "file": rel_path,
        "text": text,
        "metadata": {
            "type": "text",
            "ext": path.suffix.lower(),
        },
    }]
