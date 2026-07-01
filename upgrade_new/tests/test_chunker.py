"""Tests for metadata-preserving chunking."""

from __future__ import annotations

import pytest

from upgrade_new.src.chunker import chunk_content_units, chunk_document, chunk_documents, chunk_text


def test_chunk_text_handles_empty_text() -> None:
    assert chunk_text("", size=100, overlap=10) == []


def test_chunk_text_splits_long_paragraph_with_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(80))
    chunks = chunk_text(text, size=80, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 90 for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", size=100, overlap=100)


def test_chunk_document_preserves_pdf_metadata() -> None:
    document = {
        "id": "pdf_doc_page_1",
        "text": "Intro.\n\nSecond section with details.",
        "metadata": {
            "source_type": "pdf",
            "source_file": "lesson.pdf",
            "page_number": 1,
            "document_id": "doc123",
            "parser": "pymupdf",
        },
    }

    chunks = chunk_document(document, size=20, overlap=5)

    assert len(chunks) >= 2
    assert chunks[0]["id"] == "pdf_doc_page_1_chunk_0"
    assert chunks[0]["metadata"]["source_file"] == "lesson.pdf"
    assert chunks[0]["metadata"]["page_number"] == 1
    assert chunks[0]["metadata"]["document_id"] == "doc123"
    assert chunks[0]["metadata"]["chunk_id"] == 0
    assert chunks[1]["metadata"]["chunk_id"] == 1


def test_chunk_documents_preserves_all_metadata() -> None:
    documents = [
        {
            "id": "pdf_doc_page_2",
            "text": "Page two content.",
            "metadata": {"source_file": "lesson.pdf", "page_number": 2, "document_id": "doc123"},
        }
    ]

    chunks = chunk_documents(documents, size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Page two content."
    assert chunks[0]["metadata"]["page_number"] == 2


def test_chunk_content_units_keeps_pdf_pages_separate() -> None:
    units = [
        {
            "id": "p1b1",
            "text": "Page one text.",
            "metadata": {"source_type": "pdf", "document_id": "doc", "page_number": 1, "block_type": "text"},
        },
        {
            "id": "p2b1",
            "text": "Page two text.",
            "metadata": {"source_type": "pdf", "document_id": "doc", "page_number": 2, "block_type": "text"},
        },
    ]

    chunks = chunk_content_units(units, size=100, overlap=10)

    assert len(chunks) == 2
    assert {chunk["metadata"]["page_number"] for chunk in chunks} == {1, 2}


def test_chunk_content_units_keeps_heading_context_and_metadata() -> None:
    units = [
        {
            "id": "h1",
            "text": "Branching",
            "metadata": {"source_type": "notion", "page_id": "p", "block_type": "heading", "heading_path": "Branching"},
        },
        {
            "id": "b1",
            "text": "If else controls program flow.",
            "metadata": {"source_type": "notion", "page_id": "p", "block_type": "paragraph", "heading_path": "Branching"},
        },
    ]

    chunks = chunk_content_units(units, size=200, overlap=20)

    assert len(chunks) == 1
    assert "Branching" in chunks[0]["text"]
    assert chunks[0]["metadata"]["heading_path"] == "Branching"
    assert "paragraph" in chunks[0]["metadata"]["block_types"]


def test_chunk_content_units_does_not_repeat_heading_per_unit() -> None:
    heading = "1. Gioi thieu ve Python"
    units = [
        {
            "id": "h1",
            "text": heading,
            "metadata": {"source_type": "notion", "page_id": "p", "block_type": "heading", "heading_path": heading},
        },
        {
            "id": "b1",
            "text": "- Lich su...",
            "metadata": {"source_type": "notion", "page_id": "p", "block_type": "list", "heading_path": heading},
        },
        {
            "id": "b2",
            "text": "- Tai sao chon Python?...",
            "metadata": {"source_type": "notion", "page_id": "p", "block_type": "list", "heading_path": heading},
        },
        {
            "id": "b3",
            "text": "- Cac thu vien pho bien...",
            "metadata": {"source_type": "notion", "page_id": "p", "block_type": "list", "heading_path": heading},
        },
    ]

    chunks = chunk_content_units(units, size=1000, overlap=100)

    assert len(chunks) == 1
    assert chunks[0]["text"].count(heading) == 1
    assert "- Lich su..." in chunks[0]["text"]
    assert "- Tai sao chon Python?..." in chunks[0]["text"]
    assert "- Cac thu vien pho bien..." in chunks[0]["text"]


def test_chunk_content_units_does_not_split_small_code_block() -> None:
    code = "```python\nif score >= 5:\n    print('pass')\n```"
    units = [
        {
            "id": "code1",
            "text": code,
            "metadata": {"source_type": "notion", "page_id": "p", "block_type": "code", "code_language": "python"},
        }
    ]

    chunks = chunk_content_units(units, size=200, overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["text"] == code
    assert chunks[0]["metadata"]["code_language"] == "python"
