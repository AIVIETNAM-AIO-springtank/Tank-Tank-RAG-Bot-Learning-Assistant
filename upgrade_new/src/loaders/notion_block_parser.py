"""Convert Notion blocks into structure-aware content units."""

from __future__ import annotations

import re
from typing import Any, Callable

from upgrade_new.src import config
from upgrade_new.src.loaders.ocr_loader import extract_text_from_image


MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MARKDOWN_TODO_RE = re.compile(r"^[-+*]\s+\[([ xX])\]\s+(.+?)\s*$")
MARKDOWN_BULLET_RE = re.compile(r"^[-+*]\s+(.+?)\s*$")
MARKDOWN_NUMBERED_RE = re.compile(r"^(\d+)[.)]\s+(.+?)\s*$")


def parse_blocks_to_markdown(blocks: list[dict]) -> dict[str, Any]:
    """Backward-compatible parser that returns a markdown string."""
    parsed = parse_blocks_to_units(blocks, {})
    return {
        "markdown": "\n\n".join(unit["text"] for unit in parsed["units"] if unit.get("text")),
        "block_types": parsed["block_types"],
        "image_refs": parsed["image_refs"],
    }


def parse_blocks_to_units(
    blocks: list[dict],
    base_metadata: dict[str, Any],
    *,
    enable_ocr: bool = config.ENABLE_OCR,
    vision_fn: Callable[[str], dict[str, Any]] = extract_text_from_image,
) -> dict[str, Any]:
    """Parse Notion block trees into content units with enriched metadata."""
    state = {"heading_path": [], "number_stack": []}
    units: list[dict[str, Any]] = []
    image_refs: list[dict[str, Any]] = []
    block_types: list[str] = []

    for index, block in enumerate(blocks):
        _parse_block(
            block,
            base_metadata=base_metadata,
            units=units,
            image_refs=image_refs,
            block_types=block_types,
            state=state,
            index=index,
            enable_ocr=enable_ocr,
            vision_fn=vision_fn,
        )

    return {
        "units": units,
        "block_types": _ordered_unique(block_types),
        "image_refs": image_refs,
    }


def _parse_block(
    block: dict[str, Any],
    *,
    base_metadata: dict[str, Any],
    units: list[dict[str, Any]],
    image_refs: list[dict[str, Any]],
    block_types: list[str],
    state: dict[str, Any],
    index: int,
    enable_ocr: bool,
    vision_fn: Callable[[str], dict[str, Any]],
) -> None:
    block_type = block.get("type", "unsupported")
    block_id = block.get("id", f"block_{index}")
    block_types.append(block_type)

    if block_type == "table":
        _append_table_unit(
            block,
            base_metadata=base_metadata,
            units=units,
            block_types=block_types,
            state=state,
            block_id=block_id,
            block_index=index,
        )
        return

    text = _render_block(block, image_refs=image_refs)

    if block_type in {"heading_1", "heading_2", "heading_3"}:
        level = int(block_type[-1])
        _update_heading_path(state["heading_path"], level, text)

    if text:
        if block_type == "paragraph":
            _append_markdown_aware_units(
                text,
                base_metadata=base_metadata,
                units=units,
                state=state,
                block_id=block_id,
                block_index=index,
                notion_block_type=block_type,
            )
        else:
            metadata = _unit_metadata(
                base_metadata=base_metadata,
                block_id=block_id,
                block_index=index,
                block_type=_normalize_block_type(block_type),
                notion_block_type=block_type,
                heading_path=state["heading_path"],
            )
            if block_type == "code":
                metadata["code_language"] = block.get("code", {}).get("language", "")
            if block_type == "image":
                image_ref = image_refs[-1] if image_refs else {}
                metadata["image_url"] = image_ref.get("url", "")
                metadata["caption"] = image_ref.get("caption", "")
                if enable_ocr and metadata["image_url"]:
                    vision = vision_fn(metadata["image_url"])
                    metadata["image_ocr_text"] = "yes" if vision.get("text") else ""
                    metadata["ocr_provider"] = vision.get("provider", "gemini_vision")
                    metadata["ocr_model"] = vision.get("model", "")
                    metadata["ocr_error"] = vision.get("error", "")
                    if vision.get("text"):
                        text = f"{text}\n\nOCR/Vision:\n{vision['text']}"

            _append_unit(
                units,
                base_metadata=base_metadata,
                block_id=block_id,
                text=text,
                metadata=metadata,
            )

    children = block.get("children") or []
    for child_index, child in enumerate(children):
        _parse_block(
            child,
            base_metadata=base_metadata,
            units=units,
            image_refs=image_refs,
            block_types=block_types,
            state=state,
            index=child_index,
            enable_ocr=enable_ocr,
            vision_fn=vision_fn,
        )


def _append_table_unit(
    block: dict[str, Any],
    *,
    base_metadata: dict[str, Any],
    units: list[dict[str, Any]],
    block_types: list[str],
    state: dict[str, Any],
    block_id: str,
    block_index: int,
) -> None:
    rows = [child for child in block.get("children") or [] if child.get("type") == "table_row"]
    block_types.extend("table_row" for _ in rows)
    table_text = _render_table(rows)
    if not table_text:
        table_text = "[Table]"
    metadata = _unit_metadata(
        base_metadata=base_metadata,
        block_id=block_id,
        block_index=block_index,
        block_type="table",
        notion_block_type="table",
        heading_path=state["heading_path"],
    )
    metadata["table_row_count"] = len(rows)
    _append_unit(units, base_metadata=base_metadata, block_id=block_id, text=table_text, metadata=metadata)


def _render_table(rows: list[dict[str, Any]]) -> str:
    rendered_rows: list[list[str]] = []
    for row in rows:
        cells = row.get("table_row", {}).get("cells", [])
        rendered = [" ".join(_text_items(cell)).strip() for cell in cells]
        if any(rendered):
            rendered_rows.append(rendered)
    if not rendered_rows:
        return ""
    width = max(len(row) for row in rendered_rows)
    normalized = [row + [""] * (width - len(row)) for row in rendered_rows]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:] if len(normalized) > 1 else []
    lines = [_markdown_table_row(header), _markdown_table_row(separator)]
    lines.extend(_markdown_table_row(row) for row in body)
    return "\n".join(lines)


def _markdown_table_row(row: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"


def _render_block(block: dict[str, Any], *, image_refs: list[dict[str, Any]]) -> str:
    block_type = block.get("type", "")
    payload = block.get(block_type, {}) if block_type else {}

    if block_type == "paragraph":
        return _rich_text(payload)
    if block_type == "heading_1":
        return f"# {_rich_text(payload)}".strip()
    if block_type == "heading_2":
        return f"## {_rich_text(payload)}".strip()
    if block_type == "heading_3":
        return f"### {_rich_text(payload)}".strip()
    if block_type == "bulleted_list_item":
        return f"- {_strip_unordered_marker(_rich_text(payload))}".strip()
    if block_type == "numbered_list_item":
        return f"1. {_strip_ordered_marker(_rich_text(payload))}".strip()
    if block_type == "to_do":
        checked = "x" if payload.get("checked") else " "
        return f"- [{checked}] {_strip_todo_marker(_rich_text(payload))}".strip()
    if block_type == "quote":
        return f"> {_rich_text(payload)}".strip()
    if block_type == "code":
        language = payload.get("language", "")
        return f"```{language}\n{_rich_text(payload)}\n```".strip()
    if block_type == "equation":
        expression = payload.get("expression", "")
        return f"$$\n{expression}\n$$".strip() if expression else ""
    if block_type == "divider":
        return "---"
    if block_type == "image":
        caption = _caption(payload)
        url = _image_url(payload)
        image_refs.append({"url": url, "caption": caption, "block_id": block.get("id", "")})
        label = caption or "image block"
        return f"[Image: {label}]"
    if block_type == "table_row":
        cells = payload.get("cells", [])
        rendered_cells = [" ".join(_text_items(cell)).strip() for cell in cells]
        return "| " + " | ".join(rendered_cells) + " |" if rendered_cells else ""
    if block_type == "table":
        return "[Table]"
    return _rich_text(payload)


def _append_markdown_aware_units(
    text: str,
    *,
    base_metadata: dict[str, Any],
    units: list[dict[str, Any]],
    state: dict[str, Any],
    block_id: str,
    block_index: int,
    notion_block_type: str,
) -> None:
    """Split raw markdown pasted into paragraph blocks into structured units."""
    lines = [line.strip() for line in text.splitlines()]
    meaningful_lines = [line for line in lines if line]
    if len(meaningful_lines) <= 1:
        line = meaningful_lines[0] if meaningful_lines else text.strip()
        parsed = _parse_markdown_line(line)
        _append_parsed_markdown_line(
            parsed,
            base_metadata=base_metadata,
            units=units,
            state=state,
            block_id=block_id,
            block_index=block_index,
            notion_block_type=notion_block_type,
            line_index=None,
        )
        return

    paragraph_buffer: list[str] = []

    def flush_paragraph(line_index: int) -> None:
        if not paragraph_buffer:
            return
        paragraph = " ".join(paragraph_buffer).strip()
        paragraph_buffer.clear()
        _append_parsed_markdown_line(
            {"block_type": "paragraph", "text": paragraph, "level": None},
            base_metadata=base_metadata,
            units=units,
            state=state,
            block_id=block_id,
            block_index=block_index,
            notion_block_type=notion_block_type,
            line_index=line_index,
        )

    for line_index, line in enumerate(lines):
        if not line:
            flush_paragraph(line_index)
            continue
        parsed = _parse_markdown_line(line)
        if parsed["block_type"] == "paragraph":
            paragraph_buffer.append(parsed["text"])
            continue
        flush_paragraph(line_index)
        _append_parsed_markdown_line(
            parsed,
            base_metadata=base_metadata,
            units=units,
            state=state,
            block_id=block_id,
            block_index=block_index,
            notion_block_type=notion_block_type,
            line_index=line_index,
        )

    flush_paragraph(len(lines))


def _append_parsed_markdown_line(
    parsed: dict[str, Any],
    *,
    base_metadata: dict[str, Any],
    units: list[dict[str, Any]],
    state: dict[str, Any],
    block_id: str,
    block_index: int,
    notion_block_type: str,
    line_index: int | None,
) -> None:
    block_type = parsed["block_type"]
    text = parsed["text"].strip()
    if not text:
        return

    if block_type == "heading":
        _update_heading_path(state["heading_path"], int(parsed["level"]), text)

    metadata = _unit_metadata(
        base_metadata=base_metadata,
        block_id=block_id,
        block_index=block_index,
        block_type=block_type,
        notion_block_type=notion_block_type,
        heading_path=state["heading_path"],
    )
    if line_index is not None:
        metadata["line_index"] = line_index
    if block_type != "paragraph":
        metadata["markdown_block_type"] = block_type

    _append_unit(
        units,
        base_metadata=base_metadata,
        block_id=block_id if line_index is None else f"{block_id}_{line_index}",
        text=text,
        metadata=metadata,
    )


def _parse_markdown_line(line: str) -> dict[str, Any]:
    heading_match = MARKDOWN_HEADING_RE.match(line)
    if heading_match:
        return {
            "block_type": "heading",
            "text": heading_match.group(2).strip(),
            "level": min(len(heading_match.group(1)), 3),
        }

    todo_match = MARKDOWN_TODO_RE.match(line)
    if todo_match:
        checked = "x" if todo_match.group(1).lower() == "x" else " "
        return {"block_type": "list", "text": f"- [{checked}] {todo_match.group(2).strip()}", "level": None}

    bullet_match = MARKDOWN_BULLET_RE.match(line)
    if bullet_match:
        return {"block_type": "list", "text": f"- {bullet_match.group(1).strip()}", "level": None}

    numbered_match = MARKDOWN_NUMBERED_RE.match(line)
    if numbered_match:
        return {"block_type": "list", "text": f"{numbered_match.group(1)}. {numbered_match.group(2).strip()}", "level": None}

    return {"block_type": "paragraph", "text": line, "level": None}


def _unit_metadata(
    *,
    base_metadata: dict[str, Any],
    block_id: str,
    block_index: int,
    block_type: str,
    notion_block_type: str,
    heading_path: list[str],
) -> dict[str, Any]:
    return {
        **base_metadata,
        "source_type": "notion",
        "source_granularity": "lesson_content",
        "block_id": block_id,
        "block_index": block_index,
        "block_type": block_type,
        "notion_block_type": notion_block_type,
        "heading_path": " > ".join(heading_path),
    }


def _append_unit(
    units: list[dict[str, Any]],
    *,
    base_metadata: dict[str, Any],
    block_id: str,
    text: str,
    metadata: dict[str, Any],
) -> None:
    units.append(
        {
            "id": f"notion_{base_metadata.get('page_id', 'page')}_block_{block_id}",
            "text": text,
            "metadata": metadata,
        }
    )


def _rich_text(payload: dict[str, Any]) -> str:
    return " ".join(_text_items(payload.get("rich_text", []))).strip()


def _strip_unordered_marker(text: str) -> str:
    match = MARKDOWN_BULLET_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def _strip_ordered_marker(text: str) -> str:
    match = MARKDOWN_NUMBERED_RE.match(text.strip())
    return match.group(2).strip() if match else text.strip()


def _strip_todo_marker(text: str) -> str:
    match = MARKDOWN_TODO_RE.match(text.strip())
    return match.group(2).strip() if match else text.strip()


def _caption(payload: dict[str, Any]) -> str:
    return " ".join(_text_items(payload.get("caption", []))).strip()


def _text_items(items: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for item in items:
        plain = item.get("plain_text")
        if plain:
            texts.append(str(plain))
            continue
        text = item.get("text", {}).get("content")
        if text:
            texts.append(str(text))
    return texts


def _image_url(payload: dict[str, Any]) -> str:
    image_type = payload.get("type")
    if image_type == "external":
        return payload.get("external", {}).get("url", "")
    if image_type == "file":
        return payload.get("file", {}).get("url", "")
    return ""


def _normalize_block_type(block_type: str) -> str:
    if block_type in {"heading_1", "heading_2", "heading_3"}:
        return "heading"
    if block_type in {"bulleted_list_item", "numbered_list_item", "to_do"}:
        return "list"
    if block_type == "table_row":
        return "table"
    return block_type or "unsupported"


def _update_heading_path(path: list[str], level: int, text: str) -> None:
    clean = text.lstrip("#").strip()
    if not clean:
        return
    del path[level - 1 :]
    path.append(clean)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
