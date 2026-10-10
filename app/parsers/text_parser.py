"""Read plain text files (.txt, .log, .py). UTF-8 with latin-1 fallback."""
import logging
from pathlib import Path
log = logging.getLogger("optiseek.rag.parsers.text")

def parse_text(path, corpus_root):
    rel_path = str(path.relative_to(corpus_root))
    try:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
    except Exception as e:
        log.warning("Text read failed: %s: %s", rel_path, e); return None
    if not text.strip(): return None
    return [{"file": rel_path, "text": text, "metadata": {"type":"text","ext":path.suffix.lower()}}]
