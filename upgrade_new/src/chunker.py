"""Structure-aware chunking for PDF blocks and Notion blocks."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


Document = dict[str, Any]
ContentUnit = dict[str, Any]

DEFAULT_SCOPE_KEYS = (
    "source_type",
    "document_id",
    "page_id",
    "page_number",
    "source_granularity",
    "heading_path",
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split plain text recursively, preferring paragraphs then sentences."""
    _validate_settings(size, overlap)
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return _split_text_recursive(normalized, size=size, overlap=overlap, block_type="paragraph")


def chunk_document(document: Mapping[str, Any], size: int, overlap: int) -> list[Document]:
    """Chunk one canonical document and preserve all source metadata."""
    text = str(document.get("text") or "")
    metadata = dict(document.get("metadata") or {})
    parent_id = str(document.get("id") or _fallback_document_id(metadata))
    text_chunks = chunk_text(text, size=size, overlap=overlap)

    chunks: list[Document] = []
    for index, content in enumerate(text_chunks):
        chunk_metadata = {
            **metadata,
            "chunk_id": index,
            "parent_id": parent_id,
            "block_types": metadata.get("block_types") or "text",
        }
        chunks.append(
            {
                "id": f"{parent_id}_chunk_{index}",
                "text": content,
                "metadata": chunk_metadata,
            }
        )
    return chunks


def chunk_documents(documents: Sequence[Mapping[str, Any]], size: int, overlap: int) -> list[Document]:
    """Chunk many canonical documents and keep metadata attached."""
    chunks: list[Document] = []
    for document in documents:
        chunks.extend(chunk_document(document, size=size, overlap=overlap))
    return chunks


def chunk_content_units(units: Sequence[Mapping[str, Any]], size: int, overlap: int) -> list[Document]:
    """Chunk layout/block-aware content units without crossing source scopes."""
    return chunk_units_by_scope(units, size=size, overlap=overlap, scope_keys=DEFAULT_SCOPE_KEYS)


def chunk_units_by_scope(
    units: Sequence[Mapping[str, Any]],
    size: int,
    overlap: int,
    scope_keys: Sequence[str],
) -> list[Document]:
    """Group units by source/page/heading scope, then chunk within each group."""
    _validate_settings(size, overlap)
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)

    for unit in units:
        metadata = dict(unit.get("metadata") or {})
        key = tuple(metadata.get(scope_key, "") for scope_key in scope_keys)
        grouped[key].append(unit)

    all_chunks: list[Document] = []
    for scope_units in grouped.values():
        all_chunks.extend(_chunk_unit_group(scope_units, size=size, overlap=overlap, start_index=len(all_chunks)))
    return all_chunks


def _chunk_unit_group(units: Sequence[Mapping[str, Any]], size: int, overlap: int, start_index: int) -> list[Document]:
    chunks: list[Document] = []
    current_units: list[Mapping[str, Any]] = []
    current_length = 0
    heading_context = ""

    for unit in units:
        text = _normalize_text(str(unit.get("text") or ""))
        if not text:
            continue

        metadata = dict(unit.get("metadata") or {})
        block_type = str(metadata.get("block_type") or metadata.get("block_types") or "paragraph")

        if block_type.startswith("heading"):
            heading_context = text.lstrip("#").strip()
            if current_units:
                chunks.append(_make_chunk(current_units, start_index + len(chunks)))
                current_units = []
                current_length = 0
            continue

        metadata = {**metadata}
        if heading_context:
            metadata.setdefault("heading_context", heading_context)
        unit_text = text
        normalized_unit = {**dict(unit), "text": unit_text, "metadata": metadata}

        if len(unit_text) > size:
            if current_units:
                chunks.append(_make_chunk(current_units, start_index + len(chunks)))
                current_units = []
                current_length = 0
            for part in _split_oversized_unit(normalized_unit, size=size, overlap=overlap):
                chunks.append(_make_chunk([part], start_index + len(chunks)))
            continue

        separator = 2 if current_units else 0
        if current_length + separator + len(unit_text) <= size:
            current_units.append(normalized_unit)
            current_length += separator + len(unit_text)
            continue

        if current_units:
            chunks.append(_make_chunk(current_units, start_index + len(chunks)))
        current_units = [normalized_unit]
        current_length = len(unit_text)

    if current_units:
        chunks.append(_make_chunk(current_units, start_index + len(chunks)))

    return chunks


def _split_oversized_unit(unit: Mapping[str, Any], size: int, overlap: int) -> list[ContentUnit]:
    metadata = dict(unit.get("metadata") or {})
    block_type = str(metadata.get("block_type") or "paragraph")
    text = _normalize_text(str(unit.get("text") or ""))

    if block_type == "code":
        parts = _split_code_lines(text, size=size, overlap=overlap)
    elif block_type == "table":
        parts = _split_table_rows(text, size=size, overlap=overlap)
    elif block_type in {"equation", "image"}:
        parts = _hard_split(text, size=size, overlap=overlap)
    else:
        parts = _split_text_recursive(text, size=size, overlap=overlap, block_type=block_type)

    return [
        {
            "id": f"{unit.get('id', 'unit')}_part_{index}",
            "text": part,
            "metadata": {**metadata, "split_part": index},
        }
        for index, part in enumerate(parts)
        if part
    ]


def _make_chunk(units: Sequence[Mapping[str, Any]], chunk_index: int) -> Document:
    texts = [_normalize_text(str(unit.get("text") or "")) for unit in units]
    first = units[0]
    first_metadata = dict(first.get("metadata") or {})
    heading_context = str(first_metadata.get("heading_context") or "").strip()
    text = "\n\n".join(part for part in texts if part)
    if heading_context and not _starts_with_heading(text, heading_context):
        text = f"{heading_context}\n{text}".strip()
    block_types = _ordered_unique(
        str(dict(unit.get("metadata") or {}).get("block_type") or "text")
        for unit in units
    )
    block_ids = [
        str(dict(unit.get("metadata") or {}).get("block_id"))
        for unit in units
        if dict(unit.get("metadata") or {}).get("block_id")
    ]
    bboxes = [
        dict(unit.get("metadata") or {}).get("bbox")
        for unit in units
        if dict(unit.get("metadata") or {}).get("bbox")
    ]

    parent_id = str(first_metadata.get("parent_id") or _fallback_document_id(first_metadata))
    source_granularity = first_metadata.get("source_granularity", "content")
    chunk_id = f"{parent_id}_{source_granularity}_chunk_{chunk_index}"
    metadata = {
        **first_metadata,
        "chunk_id": chunk_index,
        "parent_id": parent_id,
        "block_types": ",".join(block_types),
    }
    if block_ids:
        metadata["block_ids"] = ",".join(block_ids)
    if bboxes:
        metadata["bbox"] = bboxes[0] if len(bboxes) == 1 else str(bboxes)

    return {"id": chunk_id, "text": text, "metadata": metadata}


def _starts_with_heading(text: str, heading: str) -> bool:
    clean_text = text.lstrip("#").strip()
    clean_heading = heading.lstrip("#").strip()
    return clean_text.startswith(clean_heading)


def _split_text_recursive(text: str, size: int, overlap: int, block_type: str) -> list[str]:
    paragraphs = _paragraphs(text)
    if len(text) <= size:
        return [text]
    if len(paragraphs) > 1:
        return _pack_segments(paragraphs, size=size, overlap=overlap)

    sentences = _sentences(text)
    if len(sentences) > 1:
        return _pack_segments(sentences, size=size, overlap=overlap)

    if block_type == "code":
        return _split_code_lines(text, size=size, overlap=overlap)
    return _hard_split(text, size=size, overlap=overlap)


def _pack_segments(segments: Sequence[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for segment in segments:
        clean = segment.strip()
        if not clean:
            continue
        if len(clean) > size:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_length = 0
            chunks.extend(_split_text_recursive(clean, size=size, overlap=overlap, block_type="paragraph"))
            continue

        separator = 2 if current else 0
        if current_length + separator + len(clean) <= size:
            current.append(clean)
            current_length += separator + len(clean)
            continue

        if current:
            previous = "\n\n".join(current).strip()
            chunks.append(previous)
            seed = previous[-overlap:].strip() if overlap else ""
            current = [seed] if seed else []
            current_length = len(seed)

        if current and current_length + 2 + len(clean) <= size:
            current.append(clean)
            current_length += 2 + len(clean)
        else:
            current = [clean]
            current_length = len(clean)

    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_code_lines(text: str, size: int, overlap: int) -> list[str]:
    return _pack_segments(text.splitlines(), size=size, overlap=overlap)


def _split_table_rows(text: str, size: int, overlap: int) -> list[str]:
    return _pack_segments(text.splitlines(), size=size, overlap=overlap)


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap if overlap else end)
    return [chunk for chunk in chunks if chunk]


def _paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    return ["\n".join(line.strip() for line in block.splitlines() if line.strip()) for block in blocks if block.strip()]


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _ordered_unique(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _validate_settings(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("chunk size must be greater than 0")
    if overlap < 0:
        raise ValueError("chunk overlap must be greater than or equal to 0")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")


def _fallback_document_id(metadata: Mapping[str, Any]) -> str:
    document_id = metadata.get("document_id") or metadata.get("page_id") or "document"
    page_number = metadata.get("page_number")
    return f"{document_id}_page_{page_number}" if page_number is not None else str(document_id)
