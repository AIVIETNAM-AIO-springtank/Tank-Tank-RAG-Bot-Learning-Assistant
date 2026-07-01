"""Tests for flexible Notion metadata and block parsing."""

from __future__ import annotations

import requests

from upgrade_new.src.chunker import chunk_content_units
from upgrade_new.src.loaders.notion_block_parser import parse_blocks_to_units
from upgrade_new.src.loaders.notion_loader import (
    _safe_request_error_detail,
    build_metadata_document,
    load_notion_lesson_from_page,
    parse_page_metadata,
    should_index_metadata_document,
)


def rich(text: str) -> list[dict]:
    return [{"type": "text", "plain_text": text, "text": {"content": text}}]


def test_parse_page_metadata_uses_title_type_and_vietnamese_aliases() -> None:
    page = {
        "id": "page123",
        "url": "https://notion.so/page123",
        "last_edited_time": "2026-06-26T00:00:00.000Z",
        "properties": {
            "Tên bài": {"type": "title", "title": rich("M01W01 - Basic Python")},
            "Tuần": {"type": "number", "number": 1},
            "Ngày học": {"type": "date", "date": {"start": "2026-06-03"}},
            "Module": {"type": "select", "select": {"name": "Module 1"}},
            "Giảng viên": {"type": "rich_text", "rich_text": rich("Dr. Quang Vinh")},
            "Chủ đề": {"type": "multi_select", "multi_select": [{"name": "Python"}, {"name": "AI"}]},
            "Tổng Hợp": {"type": "checkbox", "checkbox": True},
        },
    }

    metadata = parse_page_metadata(page)

    assert metadata["title"] == "M01W01 - Basic Python"
    assert metadata["week"] == 1
    assert metadata["date"] == "2026-06-03"
    assert metadata["module"] == "Module 1"
    assert metadata["lecturer"] == "Dr. Quang Vinh"
    assert metadata["label"] == "Python, AI"
    assert metadata["is_summary_done"] is True
    assert metadata["raw_properties_json"]


def test_build_metadata_document_shape() -> None:
    metadata = {
        "page_id": "abc",
        "title": "Basic Python",
        "week": 1,
        "date": "2026-06-03",
        "module": "Module 1",
        "lecturer": "Teacher",
        "label": "Python",
        "is_summary_done": True,
        "notion_url": "https://notion.so/abc",
        "source_type": "notion",
    }

    document = build_metadata_document(metadata)

    assert document["id"] == "notion_abc_metadata"
    assert "Basic Python" in document["text"]
    assert document["metadata"]["source_granularity"] == "lesson_metadata"
    assert document["metadata"]["block_types"] == "metadata"


def test_parse_page_metadata_reads_content_hash_property() -> None:
    page = {
        "id": "page123",
        "url": "https://notion.so/page123",
        "last_edited_time": "2026-06-26T00:00:00.000Z",
        "properties": {
            "Bai": {"type": "title", "title": rich("Basic Python")},
            "Content Hash": {"type": "rich_text", "rich_text": rich("abc123")},
        },
    }

    metadata = parse_page_metadata(page)

    assert metadata["remote_content_hash"] == "abc123"


def test_should_index_metadata_document_filters_empty_titles() -> None:
    assert should_index_metadata_document({"title": "Basic Python"}) is True
    assert should_index_metadata_document({"title": "empty", "date": "2026-06-28"}) is False
    assert should_index_metadata_document({"title": " Emty ", "module": "Module 1"}) is False
    assert should_index_metadata_document({"title": " Untitled Notion lesson "}) is False
    assert should_index_metadata_document({"title": "", "week": "", "date": "", "module": "", "lecturer": "", "label": ""}) is False


def test_load_notion_lesson_skips_empty_metadata_but_keeps_content(monkeypatch) -> None:
    page = {
        "id": "page-empty",
        "url": "https://notion.so/page-empty",
        "last_edited_time": "2026-06-28T00:00:00.000Z",
        "properties": {
            "Name": {"type": "title", "title": rich("empty")},
        },
    }
    monkeypatch.setattr(
        "upgrade_new.src.loaders.notion_loader.fetch_block_children",
        lambda page_id: [{"id": "p1", "type": "paragraph", "paragraph": {"rich_text": rich("Real lesson content.")}}],
    )

    lesson = load_notion_lesson_from_page(page)

    assert lesson["metadata_document"] is None
    assert lesson["metadata_document_skipped_empty"] is True
    assert lesson["content_units"]
    assert lesson["content_units"][0]["text"] == "Real lesson content."


def test_parse_blocks_to_units_supports_core_blocks_and_images() -> None:
    blocks = [
        {"id": "h1", "type": "heading_1", "heading_1": {"rich_text": rich("Basic Python")}},
        {"id": "p1", "type": "paragraph", "paragraph": {"rich_text": rich("Python intro.")}},
        {"id": "b1", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rich("Branching")}},
        {"id": "c1", "type": "code", "code": {"rich_text": rich("print('hi')"), "language": "python"}},
        {"id": "e1", "type": "equation", "equation": {"expression": "y = wx + b"}},
        {
            "id": "i1",
            "type": "image",
            "image": {"type": "external", "external": {"url": "https://example.com/a.png"}, "caption": rich("Slide")},
        },
    ]

    parsed = parse_blocks_to_units(blocks, {"page_id": "page1", "title": "Basic Python", "source_type": "notion"})

    texts = "\n".join(unit["text"] for unit in parsed["units"])
    assert "# Basic Python" in texts
    assert "```python" in texts
    assert "$$" in texts
    assert "[Image: Slide]" in texts
    assert parsed["image_refs"][0]["caption"] == "Slide"
    assert any(unit["metadata"]["heading_path"] == "Basic Python" for unit in parsed["units"])


def test_parse_blocks_to_units_adds_vision_text_to_images() -> None:
    blocks = [
        {
            "id": "i1",
            "type": "image",
            "image": {"type": "external", "external": {"url": "https://example.com/a.png"}, "caption": rich("Slide")},
        },
    ]

    parsed = parse_blocks_to_units(
        blocks,
        {"page_id": "page1", "title": "Basic Python", "source_type": "notion"},
        enable_ocr=True,
        vision_fn=lambda url: {
            "text": "Detected formula y = wx + b",
            "error": "",
            "provider": "gemini_vision",
            "model": "test-vision",
        },
    )

    assert "OCR/Vision" in parsed["units"][0]["text"]
    assert "Detected formula" in parsed["units"][0]["text"]
    assert parsed["units"][0]["metadata"]["ocr_provider"] == "gemini_vision"


def test_parse_blocks_to_units_renders_notion_table_as_markdown() -> None:
    blocks = [
        {
            "id": "t1",
            "type": "table",
            "table": {"table_width": 2},
            "children": [
                {"id": "r1", "type": "table_row", "table_row": {"cells": [rich("Name"), rich("Score")]}},
                {"id": "r2", "type": "table_row", "table_row": {"cells": [rich("Alice"), rich("10")]}},
            ],
        }
    ]

    parsed = parse_blocks_to_units(blocks, {"page_id": "page1", "title": "Basic Python", "source_type": "notion"})

    assert len(parsed["units"]) == 1
    assert "| Name | Score |" in parsed["units"][0]["text"]
    assert parsed["units"][0]["metadata"]["block_type"] == "table"
    assert parsed["units"][0]["metadata"]["table_row_count"] == 2


def test_parse_markdown_style_headings_and_bullets_from_paragraphs() -> None:
    text = "\n".join(
        [
            "# PHAN I: NEN TANG PYTHON",
            "## 1. Gioi thieu ve Python",
            "- Lich su: Python ra doi nam 1991.",
            "- Tai sao chon Python?",
            "- Cac thu vien pho bien:",
        ]
    )
    blocks = [{"id": "p1", "type": "paragraph", "paragraph": {"rich_text": rich(text)}}]

    parsed = parse_blocks_to_units(blocks, {"page_id": "page1", "title": "Basic Python", "source_type": "notion"})
    units = parsed["units"]

    assert [unit["metadata"]["block_type"] for unit in units] == ["heading", "heading", "list", "list", "list"]
    assert units[1]["text"] == "1. Gioi thieu ve Python"
    assert units[2]["text"] == "- Lich su: Python ra doi nam 1991."
    assert units[2]["metadata"]["heading_path"] == "PHAN I: NEN TANG PYTHON > 1. Gioi thieu ve Python"

    chunks = chunk_content_units(units, size=1000, overlap=100)

    assert len(chunks) == 1
    assert chunks[0]["text"].count("1. Gioi thieu ve Python") == 1
    assert "- Lich su: Python ra doi nam 1991." in chunks[0]["text"]
    assert chunks[0]["metadata"]["heading_path"] == "PHAN I: NEN TANG PYTHON > 1. Gioi thieu ve Python"


def test_safe_request_error_detail_redacts_notion_config(monkeypatch) -> None:
    monkeypatch.setattr("upgrade_new.src.config.NOTION_TOKEN", "secret-token")
    monkeypatch.setattr("upgrade_new.src.config.NOTION_DATABASE_ID", "database-id")

    detail = _safe_request_error_detail(
        requests.ConnectionError("failed for secret-token at /databases/database-id/query")
    )

    assert "secret-token" not in detail
    assert "database-id" not in detail
    assert detail.count("<redacted>") == 2
