"""Advanced PDF table extraction with optional pdfplumber support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from upgrade_new.src.utils.errors import PdfLoadError
from upgrade_new.src.utils.hashing import compute_file_sha256


def extract_pdf_tables(file_path: str, source_file: str | None = None) -> list[dict[str, Any]]:
    """Extract table content units from a PDF using pdfplumber when available."""
    try:
        import pdfplumber
    except Exception:
        return []

    path = Path(file_path)
    if not path.exists():
        raise PdfLoadError(f"Khong tim thay file PDF: {path}")
    if path.suffix.lower() != ".pdf":
        raise PdfLoadError("File upload phai co dinh dang PDF.")

    document_id = compute_file_sha256(path)
    source_name = source_file or path.name
    units: list[dict[str, Any]] = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages):
                tables = page.extract_tables() or []
                for table_index, rows in enumerate(tables):
                    markdown = _rows_to_markdown(rows)
                    if not markdown:
                        continue
                    page_number = page_index + 1
                    metadata = {
                        "source_type": "pdf",
                        "source_granularity": "page_table",
                        "source_file": source_name,
                        "page_number": page_number,
                        "page_count": page_count,
                        "document_id": document_id,
                        "parent_id": f"pdf_{document_id}_page_{page_number}",
                        "block_id": f"pdf_{document_id}_page_{page_number}_table_{table_index}",
                        "block_index": table_index,
                        "block_type": "table",
                        "table_index": table_index,
                        "parser": "pdfplumber",
                    }
                    units.append({"id": metadata["block_id"], "text": markdown, "metadata": metadata})
    except Exception as exc:
        raise PdfLoadError("Khong trich xuat duoc bang trong PDF bang pdfplumber.") from exc

    return units


def _rows_to_markdown(rows: list[list[Any]]) -> str:
    clean_rows = [[_cell_text(cell) for cell in row] for row in rows if row]
    clean_rows = [row for row in clean_rows if any(cell for cell in row)]
    if not clean_rows:
        return ""

    width = max(len(row) for row in clean_rows)
    normalized = [row + [""] * (width - len(row)) for row in clean_rows]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:] if len(normalized) > 1 else []
    rendered = [_markdown_row(header), _markdown_row(separator)]
    rendered.extend(_markdown_row(row) for row in body)
    return "\n".join(rendered)


def _cell_text(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def _markdown_row(row: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
