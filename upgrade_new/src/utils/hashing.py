"""Hashing helpers for file identity and Notion content sync."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_bytes_sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_notion_lesson_hash(lesson: dict[str, Any]) -> str:
    """Return a stable hash for the Notion content used by RAG.

    This is intentionally local-only. It ignores volatile sync/runtime fields
    such as last_edited_time, chunk ids and distances.
    """
    payload = _canonical_notion_lesson_payload(lesson)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return compute_bytes_sha256(data.encode("utf-8"))


def _canonical_notion_lesson_payload(lesson: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(lesson.get("metadata") or {})
    metadata_document = dict(lesson.get("metadata_document") or {})
    units = list(lesson.get("content_units") or [])

    return {
        "metadata": _pick_keys(
            metadata,
            [
                "title",
                "week",
                "date",
                "module",
                "lecturer",
                "label",
                "is_summary_done",
                "notion_url",
            ],
        ),
        "metadata_document_text": str(metadata_document.get("text") or ""),
        "content_units": [_canonical_unit(unit) for unit in units],
    }


def _canonical_unit(unit: Any) -> dict[str, Any]:
    if not isinstance(unit, dict):
        return {"text": str(unit), "metadata": {}}

    metadata = dict(unit.get("metadata") or {})
    return {
        "id": str(unit.get("id") or ""),
        "text": str(unit.get("text") or ""),
        "metadata": _pick_keys(
            metadata,
            [
                "block_id",
                "block_index",
                "block_type",
                "notion_block_type",
                "markdown_block_type",
                "heading_path",
                "code_language",
                "image_url",
                "caption",
                "line_index",
            ],
        ),
    }


def _pick_keys(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: _json_safe(source.get(key)) for key in keys if key in source}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return str(value)
