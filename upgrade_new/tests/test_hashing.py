"""Tests for content hashing helpers."""

from __future__ import annotations

from upgrade_new.src.utils.hashing import compute_notion_lesson_hash


def lesson(text: str = "Python intro.", title: str = "Basic Python") -> dict:
    return {
        "metadata": {
            "title": title,
            "week": 1,
            "date": "2026-06-03",
            "module": "Module 1",
            "lecturer": "Teacher",
            "label": "Python",
            "is_summary_done": True,
            "notion_url": "https://notion.so/page1",
            "last_edited_time": "volatile",
        },
        "metadata_document": {"text": f"Bai: {title}"},
        "content_units": [
            {
                "id": "unit1",
                "text": text,
                "metadata": {
                    "block_id": "block1",
                    "block_index": 0,
                    "block_type": "paragraph",
                    "heading_path": "Intro",
                    "chunk_id": 99,
                    "distance": 0.1,
                },
            }
        ],
    }


def test_notion_lesson_hash_is_deterministic() -> None:
    assert compute_notion_lesson_hash(lesson()) == compute_notion_lesson_hash(lesson())


def test_notion_lesson_hash_changes_when_rag_content_changes() -> None:
    assert compute_notion_lesson_hash(lesson(text="Python intro.")) != compute_notion_lesson_hash(
        lesson(text="Python advanced.")
    )


def test_notion_lesson_hash_ignores_volatile_runtime_fields() -> None:
    first = lesson()
    second = lesson()
    second["metadata"]["last_edited_time"] = "changed"
    second["content_units"][0]["metadata"]["chunk_id"] = 1
    second["content_units"][0]["metadata"]["distance"] = 0.9

    assert compute_notion_lesson_hash(first) == compute_notion_lesson_hash(second)
