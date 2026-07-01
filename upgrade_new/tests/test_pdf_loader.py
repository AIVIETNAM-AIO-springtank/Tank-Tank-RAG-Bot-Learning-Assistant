"""Tests for the PyMuPDF PDF loader."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import fitz

from upgrade_new.src.loaders.pdf_loader import load_pdf_pages, load_pdf_units
from upgrade_new.src.loaders.pdf_table_extractor import extract_pdf_tables


def test_load_pdf_pages_extracts_text_and_metadata(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Hello PDF page one")
    pdf.new_page()
    page_three = pdf.new_page()
    page_three.insert_text((72, 72), "Final page text")
    pdf.save(pdf_path)
    pdf.close()

    documents = load_pdf_pages(str(pdf_path), source_file="uploaded.pdf")

    assert len(documents) == 2
    assert documents[0]["text"].startswith("Hello PDF")
    assert documents[0]["metadata"]["source_type"] == "pdf"
    assert documents[0]["metadata"]["source_file"] == "uploaded.pdf"
    assert documents[0]["metadata"]["page_number"] == 1
    assert documents[0]["metadata"]["page_count"] == 3
    assert documents[0]["metadata"]["parser"] == "pymupdf"
    assert documents[1]["metadata"]["page_number"] == 3


def test_load_pdf_units_extracts_block_metadata(tmp_path) -> None:
    pdf_path = tmp_path / "blocks.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "First layout block")
    page.insert_text((72, 120), "Second layout block")
    pdf.save(pdf_path)
    pdf.close()

    units = load_pdf_units(str(pdf_path), source_file="blocks.pdf")

    assert units
    assert units[0]["metadata"]["source_type"] == "pdf"
    assert units[0]["metadata"]["source_granularity"] == "page_block"
    assert units[0]["metadata"]["source_file"] == "blocks.pdf"
    assert units[0]["metadata"]["page_number"] == 1
    assert units[0]["metadata"]["block_type"] == "text"
    assert units[0]["metadata"]["bbox"]


def test_load_pdf_units_uses_vision_for_scanned_page(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(pdf_path)
    pdf.close()

    monkeypatch.setattr(
        "upgrade_new.src.loaders.pdf_loader.extract_text_from_image_bytes",
        lambda image_bytes, mime_type="image/png": {
            "text": "OCR text from scanned page",
            "error": "",
            "provider": "gemini_vision",
            "model": "test-vision",
        },
    )

    units = load_pdf_units(str(pdf_path), source_file="scan.pdf", enable_ocr=True)

    assert len(units) == 1
    assert units[0]["text"] == "OCR text from scanned page"
    assert units[0]["metadata"]["block_type"] == "image"
    assert units[0]["metadata"]["ocr_provider"] == "gemini_vision"


def test_extract_pdf_tables_returns_markdown_units(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "table.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

    class FakePdf:
        pages = [
            SimpleNamespace(extract_tables=lambda: [[["Name", "Score"], ["Alice", "10"]]]),
        ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    fake_pdfplumber = SimpleNamespace(open=lambda path: FakePdf())
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)

    units = extract_pdf_tables(str(pdf_path), source_file="table.pdf")

    assert len(units) == 1
    assert "| Name | Score |" in units[0]["text"]
    assert units[0]["metadata"]["block_type"] == "table"
    assert units[0]["metadata"]["parser"] == "pdfplumber"
