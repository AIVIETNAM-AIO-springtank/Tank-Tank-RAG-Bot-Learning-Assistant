"""Small helpers for extracting image references from Notion/PDF-like blocks."""

from __future__ import annotations

from typing import Any


def extract_image_reference(block: dict[str, Any]) -> dict[str, str]:
    """Return URL/caption metadata for a Notion image block when available."""
    block_type = block.get("type", "")
    payload = block.get(block_type, {}) if block_type else {}
    caption = " ".join(item.get("plain_text", "") for item in payload.get("caption", [])).strip()
    image_type = payload.get("type")
    if image_type == "external":
        url = payload.get("external", {}).get("url", "")
    elif image_type == "file":
        url = payload.get("file", {}).get("url", "")
    else:
        url = ""
    return {"url": url, "caption": caption, "block_id": str(block.get("id", ""))}
