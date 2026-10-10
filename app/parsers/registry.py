"""Dispatch to the right parser by file extension. Defensive: never raises."""
import logging, os
from pathlib import Path

log = logging.getLogger("optiseek.rag.parsers")

EXTENSION_MAP = {
    ".pdf": "pdf_parser", ".docx": "docx_parser", ".xlsx": "xlsx_parser",
    ".csv": "csv_parser", ".txt": "text_parser", ".log": "text_parser",
    ".py": "text_parser", ".png": "image_parser", ".jpg": "image_parser",
    ".jpeg": "image_parser", ".tif": "image_parser", ".tiff": "image_parser",
}

def parse_file(path, corpus_root, vlm=None):
    ext = path.suffix.lower()
    if ext not in EXTENSION_MAP:
        log.debug("Skipping unknown type: %s", path); return None
    try:
        if not os.access(path, os.R_OK):
            log.warning("No read permission: %s", path); return None
    except OSError:
        return None
    parser_name = EXTENSION_MAP[ext]
    try:
        if parser_name == "pdf_parser":
            from .pdf_parser import parse_pdf; docs = parse_pdf(path, corpus_root)
        elif parser_name == "docx_parser":
            from .docx_parser import parse_docx; docs = parse_docx(path, corpus_root)
        elif parser_name == "xlsx_parser":
            from .xlsx_parser import parse_xlsx; docs = parse_xlsx(path, corpus_root)
        elif parser_name == "csv_parser":
            from .csv_parser import parse_csv; docs = parse_csv(path, corpus_root)
        elif parser_name == "text_parser":
            from .text_parser import parse_text; docs = parse_text(path, corpus_root)
        elif parser_name == "image_parser":
            from .image_parser import parse_image; docs = parse_image(path, corpus_root, vlm)
        else:
            return None
        return docs
    except PermissionError:
        log.warning("Permission denied: %s", path); return None
    except Exception as e:
        log.warning("Parse failed for %s: %s: %s", path, type(e).__name__, e); return None

def parse_corpus(corpus_root, vlm=None):
    all_docs = []
    for dirpath, dirnames, filenames in os.walk(corpus_root):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            docs = parse_file(filepath, corpus_root, vlm)
            if docs: all_docs.extend(docs)
    return all_docs
