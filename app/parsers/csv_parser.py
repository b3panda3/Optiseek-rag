"""Extract text from CSV files. Rows as tab-separated lines."""
import csv, logging
from pathlib import Path
log = logging.getLogger("optiseek.rag.parsers.csv")

def parse_csv(path, corpus_root):
    rel_path = str(path.relative_to(corpus_root))
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
    except Exception as e:
        log.warning("CSV open failed: %s: %s", rel_path, e); return None
    if not rows: return None
    lines = ["\t".join(row) for row in rows if row]
    full_text = "\n".join(lines)
    if not full_text.strip(): return None
    return [{"file": rel_path, "text": full_text, "metadata": {"type":"csv","rows":len(rows)}}]
