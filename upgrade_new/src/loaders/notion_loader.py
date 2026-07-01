"""Flexible Notion database/page loader for AIO lesson content."""

from __future__ import annotations

import json
import unicodedata
from typing import Any

import requests

from upgrade_new.src import config
from upgrade_new.src.loaders.notion_block_parser import parse_blocks_to_units
from upgrade_new.src.utils.errors import ConfigError, NotionError

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
EMPTY_METADATA_TITLES = {"", "empty", "emty", "untitled notion lesson"}

FIELD_ALIASES = {
    "week": {"week", "tuan", "tuần", "w"},
    "date": {"date", "ngay", "ngày", "ngay hoc", "ngày học"},
    "module": {"module", "mod", "mo dun", "mô đun", "mô-đun"},
    "lecturer": {"lecturer", "giang vien", "giảng viên", "teacher", "instructor"},
    "label": {"label", "labels", "tag", "tags", "chu de", "chủ đề"},
    "is_summary_done": {"tong hop", "tổng hợp", "summary", "done", "completed", "is_summary_done"},
}


def load_notion_lessons() -> list[dict[str, Any]]:
    """Load Notion lessons as metadata documents plus structure-aware content units."""
    pages = query_database_pages()
    return [load_notion_lesson_from_page(page) for page in pages]


def load_notion_lesson_from_page(page: dict[str, Any]) -> dict[str, Any]:
    """Load one Notion database row and its child page blocks as a lesson."""
    metadata = parse_page_metadata(page)
    blocks = fetch_block_children(page["id"])
    parsed = parse_blocks_to_units(blocks, metadata)
    should_index_metadata = should_index_metadata_document(metadata)
    metadata_document = build_metadata_document(metadata) if should_index_metadata else None
    return {
        "page_id": metadata["page_id"],
        "metadata": metadata,
        "metadata_document": metadata_document,
        "metadata_document_skipped_empty": not should_index_metadata,
        "content_units": parsed["units"],
        "block_types": parsed["block_types"],
        "image_refs": parsed["image_refs"],
    }


def query_database_pages() -> list[dict[str, Any]]:
    """Query all pages from the configured Notion database."""
    if not config.NOTION_TOKEN:
        raise ConfigError("Missing NOTION_TOKEN. Add it to environment, .env or Streamlit secrets.")
    if not config.NOTION_DATABASE_ID:
        raise ConfigError("Missing NOTION_DATABASE_ID. Add it to environment, .env or Streamlit secrets.")

    pages: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        payload: dict[str, Any] = {"page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        data = _notion_request("POST", f"/databases/{config.NOTION_DATABASE_ID}/query", json=payload)
        pages.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return pages


def fetch_block_children(block_id: str) -> list[dict[str, Any]]:
    """Fetch block children recursively from Notion."""
    children = _fetch_direct_children(block_id)
    for block in children:
        if block.get("has_children"):
            block["children"] = fetch_block_children(block["id"])
    return children


def parse_page_metadata(page: dict[str, Any]) -> dict[str, Any]:
    """Extract lesson metadata with flexible property aliases."""
    properties = page.get("properties", {})
    title = _title_from_properties(properties) or "Untitled Notion lesson"
    normalized_name_map = {_normalize_key(name): name for name in properties}

    metadata = {
        "source_type": "notion",
        "page_id": page.get("id", ""),
        "title": title,
        "week": _extract_alias(properties, normalized_name_map, "week"),
        "date": _extract_alias(properties, normalized_name_map, "date"),
        "module": _extract_alias(properties, normalized_name_map, "module"),
        "lecturer": _extract_alias(properties, normalized_name_map, "lecturer"),
        "label": _extract_alias(properties, normalized_name_map, "label"),
        "is_summary_done": bool(_extract_alias(properties, normalized_name_map, "is_summary_done")),
        "notion_url": page.get("url", ""),
        "last_edited_time": page.get("last_edited_time", ""),
        "remote_content_hash": _content_hash_from_properties(properties),
        "raw_properties_json": json.dumps(_plain_properties(properties), ensure_ascii=False),
    }
    return metadata


def build_metadata_document(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build one searchable document from a Notion database row."""
    page_id = metadata.get("page_id", "")
    lines = [
        f"Bài: {metadata.get('title') or ''}",
        f"Week: {metadata.get('week') or ''}",
        f"Date: {metadata.get('date') or ''}",
        f"Module: {metadata.get('module') or ''}",
        f"Lecturer: {metadata.get('lecturer') or ''}",
        f"Label: {metadata.get('label') or ''}",
        f"Tổng hợp: {metadata.get('is_summary_done')}",
    ]
    return {
        "id": f"notion_{page_id}_metadata",
        "text": "\n".join(lines),
        "metadata": {
            **metadata,
            "source_granularity": "lesson_metadata",
            "parent_id": f"notion_{page_id}_metadata",
            "block_types": "metadata",
        },
    }


def should_index_metadata_document(metadata: dict[str, Any]) -> bool:
    """Return whether a Notion metadata-only document is useful enough to index."""
    title = _normalized_metadata_value(metadata.get("title"))
    if title in EMPTY_METADATA_TITLES:
        return False

    useful_fields = ["title", "week", "date", "module", "lecturer", "label"]
    return any(_normalized_metadata_value(metadata.get(field)) for field in useful_fields)


def read_remote_content_hash(metadata: dict[str, Any]) -> str:
    """Return the Content Hash value currently stored on a Notion row."""
    return str(metadata.get("remote_content_hash") or "").strip()


def write_content_hash_to_notion(page_id: str, content_hash: str) -> dict[str, Any]:
    """Write the local content hash into the configured Notion property."""
    if not page_id or not content_hash:
        return {"ok": False, "error": "missing page_id or content_hash"}
    property_name = config.NOTION_CONTENT_HASH_PROPERTY
    payload = {
        "properties": {
            property_name: {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": content_hash},
                    }
                ]
            }
        }
    }
    _notion_request("PATCH", f"/pages/{page_id}", json=payload)
    return {"ok": True, "property": property_name, "content_hash": content_hash}


def _fetch_direct_children(block_id: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        data = _notion_request("GET", f"/blocks/{block_id}/children", params=params)
        blocks.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return blocks


def _notion_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {config.NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    try:
        response = requests.request(
            method,
            f"{NOTION_API_BASE}{path}",
            headers=headers,
            timeout=config.REQUEST_TIMEOUT,
            **kwargs,
        )
    except requests.RequestException as exc:
        detail = _safe_request_error_detail(exc)
        raise NotionError(
            f"Khong goi duoc Notion API ({exc.__class__.__name__}). "
            f"Hay kiem tra network/proxy/integration token. Chi tiet: {detail}"
        ) from exc

    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        raise NotionError(f"Notion API lỗi {response.status_code}: {detail}")
    return response.json()


def _extract_alias(properties: dict[str, Any], name_map: dict[str, str], field: str) -> Any:
    for alias in FIELD_ALIASES[field]:
        property_name = name_map.get(_normalize_key(alias))
        if property_name:
            return _plain_property_value(properties[property_name])
    return None


def _title_from_properties(properties: dict[str, Any]) -> str:
    for prop in properties.values():
        if prop.get("type") == "title":
            return _plain_property_value(prop) or ""
    return ""


def _content_hash_from_properties(properties: dict[str, Any]) -> str:
    prop = properties.get(config.NOTION_CONTENT_HASH_PROPERTY)
    if not prop:
        return ""
    return str(_plain_property_value(prop) or "").strip()


def _plain_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {name: _plain_property_value(prop) for name, prop in properties.items()}


def _plain_property_value(prop: dict[str, Any]) -> Any:
    prop_type = prop.get("type")
    value = prop.get(prop_type) if prop_type else None
    if prop_type in {"title", "rich_text"}:
        return " ".join(_plain_text_items(value or []))
    if prop_type == "number":
        return value
    if prop_type == "date":
        return (value or {}).get("start", "") if isinstance(value, dict) else ""
    if prop_type == "select":
        return (value or {}).get("name", "") if isinstance(value, dict) else ""
    if prop_type == "multi_select":
        return ", ".join(item.get("name", "") for item in value or [])
    if prop_type == "checkbox":
        return bool(value)
    if prop_type == "url":
        return value or ""
    if prop_type == "email":
        return value or ""
    if prop_type == "phone_number":
        return value or ""
    if prop_type == "status":
        return (value or {}).get("name", "") if isinstance(value, dict) else ""
    if prop_type == "people":
        return ", ".join(person.get("name", "") for person in value or [])
    if prop_type == "formula":
        return _plain_formula(value or {})
    if prop_type == "rollup":
        return str(value or "")
    return str(value or "")


def _plain_formula(value: dict[str, Any]) -> Any:
    formula_type = value.get("type")
    return value.get(formula_type) if formula_type else ""


def _normalized_metadata_value(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_request_error_detail(error: requests.RequestException) -> str:
    detail = str(error).replace("\n", " ").strip()
    for secret in (config.NOTION_TOKEN, config.NOTION_DATABASE_ID):
        if secret:
            detail = detail.replace(secret, "<redacted>")
    return detail[:300] if detail else error.__class__.__name__


def _plain_text_items(items: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for item in items:
        if item.get("plain_text"):
            texts.append(item["plain_text"])
        elif item.get("text", {}).get("content"):
            texts.append(item["text"]["content"])
    return texts


def _normalize_key(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("_", " ").replace("-", " ")
    return " ".join(value.split())
