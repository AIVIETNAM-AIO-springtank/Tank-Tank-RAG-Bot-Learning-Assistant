"""PyMuPDF-based PDF loader with page and layout-block metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from upgrade_new.src import config
from upgrade_new.src.loaders.ocr_loader import extract_text_from_image_bytes
from upgrade_new.src.loaders.pdf_table_extractor import extract_pdf_tables
from upgrade_new.src.utils.errors import PdfLoadError
from upgrade_new.src.utils.hashing import compute_file_sha256


def load_pdf_pages(file_path: str, source_file: str | None = None) -> list[dict[str, Any]]:
    """Load a PDF into page-level canonical documents.

    Empty pages are skipped. This compatibility API only returns text-layer
    page documents; use ``load_pdf_units(enable_ocr=True)`` for OCR/Vision.
    """
    path = Path(file_path)
    if not path.exists():
        raise PdfLoadError(f"Không tìm thấy file PDF: {path}")
    if path.suffix.lower() != ".pdf":
        raise PdfLoadError("File upload phải có định dạng PDF.")

    document_id = compute_file_sha256(path)
    source_name = source_file or path.name

    try:
        pdf = fitz.open(path)
    except Exception as exc:
        raise PdfLoadError("Không mở được PDF. Hãy kiểm tra file và thử lại.") from exc

    try:
        page_count = pdf.page_count
        if page_count == 0:
            raise PdfLoadError("PDF không có trang nào để xử lý.")

        documents: list[dict[str, Any]] = []
        for page_index in range(page_count):
            page = pdf.load_page(page_index)
            text = _extract_page_text_from_dict(page)
            if not text:
                continue

            page_number = page_index + 1
            metadata = {
                "source_type": "pdf",
                "source_file": source_name,
                "page_number": page_number,
                "page_count": page_count,
                "document_id": document_id,
                "parser": "pymupdf",
            }
            documents.append(
                {
                    "id": f"pdf_{document_id}_page_{page_number}",
                    "text": text,
                    "metadata": metadata,
                }
            )

        if not documents:
            raise PdfLoadError(
                "PDF không có text layer trích xuất được. Hãy dùng luồng load_pdf_units với OCR/Vision nếu đây là file scan."
            )
        return documents
    finally:
        pdf.close()


def load_pdf_units(
    file_path: str,
    source_file: str | None = None,
    *,
    enable_ocr: bool = config.ENABLE_OCR,
) -> list[dict[str, Any]]:
    """Load a PDF into layout/block-aware content units.

    Each text block stays page-local and carries bbox/block metadata so chunking
    can remain page-aware while preserving layout evidence for citations/debug.
    """
    path = Path(file_path)
    if not path.exists():
        raise PdfLoadError(f"Không tìm thấy file PDF: {path}")
    if path.suffix.lower() != ".pdf":
        raise PdfLoadError("File upload phải có định dạng PDF.")

    document_id = compute_file_sha256(path)
    source_name = source_file or path.name

    try:
        pdf = fitz.open(path)
    except Exception as exc:
        raise PdfLoadError("Không mở được PDF. Hãy kiểm tra file và thử lại.") from exc

    try:
        page_count = pdf.page_count
        if page_count == 0:
            raise PdfLoadError("PDF không có trang nào để xử lý.")

        units: list[dict[str, Any]] = []
        vision_count = 0
        for page_index in range(page_count):
            page = pdf.load_page(page_index)
            page_number = page_index + 1
            page_dict = page.get_text("dict") or {}
            blocks = page_dict.get("blocks", [])
            page_text_units = 0

            for block_index, block in enumerate(blocks):
                block_type = block.get("type")
                if block_type == 1 and enable_ocr and vision_count < config.VISION_MAX_IMAGES_PER_DOC:
                    unit = _ocr_image_block(
                        block,
                        document_id=document_id,
                        source_name=source_name,
                        page_number=page_number,
                        page_count=page_count,
                        image_index=vision_count,
                    )
                    vision_count += 1
                    if unit:
                        units.append(unit)
                    continue
                if block_type != 0:
                    continue
                text = _block_text(block)
                if not text:
                    continue
                page_text_units += 1

                metadata = {
                    "source_type": "pdf",
                    "source_granularity": "page_block",
                    "source_file": source_name,
                    "page_number": page_number,
                    "page_count": page_count,
                    "document_id": document_id,
                    "parent_id": f"pdf_{document_id}_page_{page_number}",
                    "block_id": f"pdf_{document_id}_page_{page_number}_block_{block_index}",
                    "block_index": block_index,
                    "block_type": "text",
                    "bbox": _bbox(block.get("bbox")),
                    "parser": "pymupdf",
                }
                units.append(
                    {
                        "id": metadata["block_id"],
                        "text": text,
                        "metadata": metadata,
                    }
                )

            if enable_ocr and page_text_units == 0 and vision_count < config.VISION_MAX_IMAGES_PER_DOC:
                unit = _ocr_rendered_page(
                    page,
                    document_id=document_id,
                    source_name=source_name,
                    page_number=page_number,
                    page_count=page_count,
                    image_index=vision_count,
                )
                vision_count += 1
                if unit:
                    units.append(unit)

        units.extend(extract_pdf_tables(file_path, source_file=source_name))

        if not units:
            raise PdfLoadError(
                "PDF không có text, OCR/Vision, hoặc table content có thể index."
            )
        return units
    finally:
        pdf.close()


def _extract_page_text_from_dict(page: Any) -> str:
    page_dict = page.get_text("dict") or {}
    blocks = [_block_text(block) for block in page_dict.get("blocks", []) if block.get("type") == 0]
    return "\n\n".join(block for block in blocks if block).strip()


def _block_text(block: dict[str, Any]) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        line_text = "".join(spans).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def _bbox(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return ""
    return ",".join(f"{float(item):.2f}" for item in value)


def _ocr_image_block(
    block: dict[str, Any],
    *,
    document_id: str,
    source_name: str,
    page_number: int,
    page_count: int,
    image_index: int,
) -> dict[str, Any] | None:
    image_bytes = block.get("image")
    if not image_bytes:
        return None
    extension = str(block.get("ext") or "png").lstrip(".").lower()
    mime_type = f"image/{'jpeg' if extension in {'jpg', 'jpeg'} else extension}"
    result = extract_text_from_image_bytes(image_bytes, mime_type=mime_type)
    text = str(result.get("text") or "").strip()
    error = str(result.get("error") or "")
    if not text and not error:
        return None
    metadata = _ocr_metadata(
        document_id=document_id,
        source_name=source_name,
        page_number=page_number,
        page_count=page_count,
        image_index=image_index,
        block_id=f"pdf_{document_id}_page_{page_number}_image_{image_index}",
        bbox=_bbox(block.get("bbox")),
        unit_kind="image",
        error=error,
    )
    return {"id": metadata["block_id"], "text": text or "[Image OCR failed]", "metadata": metadata}


def _ocr_rendered_page(
    page: Any,
    *,
    document_id: str,
    source_name: str,
    page_number: int,
    page_count: int,
    image_index: int,
) -> dict[str, Any] | None:
    zoom = max(1.0, float(config.VISION_IMAGE_DPI) / 72.0)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    result = extract_text_from_image_bytes(pixmap.tobytes("png"), mime_type="image/png")
    text = str(result.get("text") or "").strip()
    error = str(result.get("error") or "")
    if not text and not error:
        return None
    block_id = f"pdf_{document_id}_page_{page_number}_ocr_{image_index}"
    metadata = _ocr_metadata(
        document_id=document_id,
        source_name=source_name,
        page_number=page_number,
        page_count=page_count,
        image_index=image_index,
        block_id=block_id,
        bbox="",
        unit_kind="page_ocr",
        error=error,
    )
    return {"id": block_id, "text": text or "[Page OCR failed]", "metadata": metadata}


def _ocr_metadata(
    *,
    document_id: str,
    source_name: str,
    page_number: int,
    page_count: int,
    image_index: int,
    block_id: str,
    bbox: str,
    unit_kind: str,
    error: str,
) -> dict[str, Any]:
    return {
        "source_type": "pdf",
        "source_granularity": "page_image",
        "source_file": source_name,
        "page_number": page_number,
        "page_count": page_count,
        "document_id": document_id,
        "parent_id": f"pdf_{document_id}_page_{page_number}",
        "block_id": block_id,
        "block_index": image_index,
        "block_type": "image",
        "image_index": image_index,
        "image_ocr_text": "yes" if not error else "",
        "ocr_provider": "gemini_vision",
        "ocr_error": error,
        "ocr_unit_kind": unit_kind,
        "bbox": bbox,
        "parser": "pymupdf+gemini_vision",
    }
