"""xlsx_parser.py — Extract data from Excel spreadsheets.

Uses openpyxl to read all sheets. Each row is serialized as a tab-separated
line, with the sheet name prefixed. This preserves the tabular structure
so the LLM can reason about columns (e.g., "part_number", "compatible_with").
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("optiseek.rag.parsers.xlsx")


def parse_xlsx(path: Path, corpus_root: Path) -> list[dict] | None:
    """Extract text from an .xlsx file. Returns one document per sheet."""
    from openpyxl import load_workbook

    rel_path = str(path.relative_to(corpus_root))

    try:
        wb = load_workbook(str(path), read_only=True, data_only=True)
    except Exception as e:
        log.warning("XLSX open failed: %s: %s", rel_path, e)
        return None

    documents = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_text = []

        for row in ws.iter_rows(values_only=True):
            # Convert each cell to string, skip None
            cells = []
            for cell in row:
                if cell is not None:
                    cells.append(str(cell))
            if cells:
                rows_text.append("\t".join(cells))

        if rows_text:
            documents.append({
                "file": rel_path,
                "text": f"Sheet: {sheet_name}\n" + "\n".join(rows_text),
                "metadata": {
                    "type": "xlsx",
                    "sheet": sheet_name,
                    "rows": len(rows_text),
                },
            })

    wb.close()
    return documents if documents else None
