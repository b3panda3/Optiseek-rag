"""Extract data from Excel spreadsheets. Multi-sheet, tab-separated rows."""
import logging
from pathlib import Path
log = logging.getLogger("optiseek.rag.parsers.xlsx")

def parse_xlsx(path, corpus_root):
    from openpyxl import load_workbook
    rel_path = str(path.relative_to(corpus_root))
    try:
        wb = load_workbook(str(path), read_only=True, data_only=True)
    except Exception as e:
        log.warning("XLSX open failed: %s: %s", rel_path, e); return None
    documents = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_text = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells: rows_text.append("\t".join(cells))
        if rows_text:
            documents.append({"file": rel_path, "text": f"Sheet: {sheet_name}\n" + "\n".join(rows_text), "metadata": {"type":"xlsx","sheet":sheet_name,"rows":len(rows_text)}})
    wb.close()
    return documents if documents else None
