"""csv_parser.py — Extract text from CSV files.

Each row is serialized as a tab-separated line. The header row is included
so the LLM can identify columns (e.g., ticket, fixed_in).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger("optiseek.rag.parsers.csv")


def parse_csv(path: Path, corpus_root: Path) -> list[dict] | None:
    """Extract text from a .csv file."""
    rel_path = str(path.relative_to(corpus_root))

    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        log.warning("CSV open failed: %s: %s", rel_path, e)
        return None

    if not rows:
        return None

    lines = ["\t".join(row) for row in rows if row]
    full_text = "\n".join(lines)

    if not full_text.strip():
        return None

    return [{
        "file": rel_path,
        "text": full_text,
        "metadata": {
            "type": "csv",
            "rows": len(rows),
        },
    }]
