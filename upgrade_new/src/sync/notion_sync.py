"""Manual and incremental Notion sync orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from upgrade_new.src import config
from upgrade_new.src.chunker import chunk_content_units
from upgrade_new.src.embeddings import embed_documents
from upgrade_new.src.loaders.notion_loader import (
    load_notion_lesson_from_page,
    load_notion_lessons,
    parse_page_metadata,
    query_database_pages,
    read_remote_content_hash,
    should_index_metadata_document,
    write_content_hash_to_notion,
)
from upgrade_new.src.sync.sync_state import SyncStateStore, now_utc_iso
from upgrade_new.src.utils.hashing import compute_notion_lesson_hash
from upgrade_new.src.vector_store import VectorStore


VALID_SYNC_MODES = {"incremental", "full"}


def sync_notion(
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    *,
    sync_mode: str = "incremental",
    lessons: list[dict[str, Any]] | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] = embed_documents,
    vector_store: VectorStore | None = None,
    sync_state_store: SyncStateStore | None = None,
    query_pages_fn: Callable[[], list[dict[str, Any]]] = query_database_pages,
    page_loader_fn: Callable[[dict[str, Any]], dict[str, Any]] = load_notion_lesson_from_page,
    hash_writer_fn: Callable[[str, str], dict[str, Any]] = write_content_hash_to_notion,
) -> dict[str, Any]:
    """Sync Notion content into ChromaDB.

    ``incremental`` updates only new/changed/removed pages based on local state.
    ``full`` rebuilds all Notion documents while preserving non-Notion chunks.
    """
    if sync_mode not in VALID_SYNC_MODES:
        raise ValueError(f"sync_mode must be one of {sorted(VALID_SYNC_MODES)}")

    store = vector_store or VectorStore()
    state_store = sync_state_store or SyncStateStore()

    if sync_mode == "full":
        return _sync_full(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            lessons=lessons,
            embed_fn=embed_fn,
            store=store,
            state_store=state_store,
            hash_writer_fn=hash_writer_fn,
        )

    return _sync_incremental(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        lessons=lessons,
        embed_fn=embed_fn,
        store=store,
        state_store=state_store,
        query_pages_fn=query_pages_fn,
        page_loader_fn=page_loader_fn,
        hash_writer_fn=hash_writer_fn,
    )


def _sync_full(
    *,
    chunk_size: int,
    chunk_overlap: int,
    lessons: list[dict[str, Any]] | None,
    embed_fn: Callable[[list[str]], list[list[float]]],
    store: VectorStore,
    state_store: SyncStateStore,
    hash_writer_fn: Callable[[str, str], dict[str, Any]],
) -> dict[str, Any]:
    loaded_lessons = lessons if lessons is not None else load_notion_lessons()
    summary = _empty_summary(sync_mode="full", pages_seen=len(loaded_lessons))

    state = state_store.load()
    state["notion_pages"] = {}

    documents_to_index: list[dict[str, Any]] = []
    lessons_to_write_hash: list[dict[str, Any]] = []
    for lesson in loaded_lessons:
        prepared = _prepare_lesson_documents(lesson, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_to_index.extend(prepared["documents"])
        _accumulate_document_counts(summary, prepared)
        _record_indexed_lesson(state, lesson, prepared["documents"], status="full")
        if prepared["documents"]:
            lessons_to_write_hash.append(lesson)

    store.delete_by_source_type("notion")
    if documents_to_index:
        embeddings = embed_fn([document["text"] for document in documents_to_index])
        store.add_documents(documents_to_index, embeddings)

    for lesson in lessons_to_write_hash:
        _maybe_write_notion_hash(lesson, state=state, summary=summary, hash_writer_fn=hash_writer_fn)

    state_store.save(state)
    summary["pages_new"] = len(loaded_lessons)
    summary["chunks_indexed"] = len(documents_to_index)
    summary["collection_count"] = store.count()
    return summary


def _sync_incremental(
    *,
    chunk_size: int,
    chunk_overlap: int,
    lessons: list[dict[str, Any]] | None,
    embed_fn: Callable[[list[str]], list[list[float]]],
    store: VectorStore,
    state_store: SyncStateStore,
    query_pages_fn: Callable[[], list[dict[str, Any]]],
    page_loader_fn: Callable[[dict[str, Any]], dict[str, Any]],
    hash_writer_fn: Callable[[str, str], dict[str, Any]],
) -> dict[str, Any]:
    state = state_store.load()
    summary = _empty_summary(sync_mode="incremental")

    if lessons is not None:
        current_page_ids = {_lesson_page_id(lesson) for lesson in lessons if _lesson_page_id(lesson)}
        summary["pages_seen"] = len(lessons)
        _remove_missing_pages(state, current_page_ids, store, summary)
        for lesson in lessons:
            _sync_loaded_lesson(
                lesson,
                state=state,
                store=store,
                embed_fn=embed_fn,
                summary=summary,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                hash_writer_fn=hash_writer_fn,
            )
        state_store.save(state)
        summary["collection_count"] = store.count()
        return summary

    pages = query_pages_fn()
    summary["pages_seen"] = len(pages)
    current_page_ids: set[str] = set()

    for page in pages:
        metadata = parse_page_metadata(page)
        page_id = str(metadata.get("page_id") or "")
        if not page_id:
            continue
        current_page_ids.add(page_id)

        existing = _state_pages(state).get(page_id)
        if _can_skip_by_remote_hash(existing, metadata) or _can_skip_by_last_edited(existing, metadata):
            summary["pages_skipped"] += 1
            continue

        lesson = page_loader_fn(page)
        _sync_loaded_lesson(
            lesson,
            state=state,
            store=store,
            embed_fn=embed_fn,
            summary=summary,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            hash_writer_fn=hash_writer_fn,
        )

    _remove_missing_pages(state, current_page_ids, store, summary)
    state_store.save(state)
    summary["collection_count"] = store.count()
    return summary


def _sync_loaded_lesson(
    lesson: dict[str, Any],
    *,
    state: dict[str, Any],
    store: VectorStore,
    embed_fn: Callable[[list[str]], list[list[float]]],
    summary: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    hash_writer_fn: Callable[[str, str], dict[str, Any]],
) -> None:
    page_id = _lesson_page_id(lesson)
    if not page_id:
        return

    existing = _state_pages(state).get(page_id)
    content_hash = compute_notion_lesson_hash(lesson)
    if existing and existing.get("content_hash") == content_hash:
        summary["pages_hash_unchanged"] += 1
        _record_hash_unchanged_lesson(state, lesson, existing, content_hash)
        return

    prepared = _prepare_lesson_documents(lesson, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store.delete_by_page_id(page_id)
    if prepared["documents"]:
        embeddings = embed_fn([document["text"] for document in prepared["documents"]])
        store.add_documents(prepared["documents"], embeddings)

    if existing:
        summary["pages_changed"] += 1
    else:
        summary["pages_new"] += 1
    _accumulate_document_counts(summary, prepared)
    summary["chunks_indexed"] += len(prepared["documents"])
    _record_indexed_lesson(state, lesson, prepared["documents"], status="indexed", content_hash=content_hash)
    _maybe_write_notion_hash(lesson, state=state, summary=summary, hash_writer_fn=hash_writer_fn)


def _prepare_lesson_documents(lesson: dict[str, Any], *, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    metadata_document = lesson.get("metadata_document")
    metadata_documents = [metadata_document] if _should_index_lesson_metadata_document(lesson, metadata_document) else []
    metadata_documents_skipped_empty = 1 if lesson.get("metadata_document_skipped_empty") or (metadata_document and not metadata_documents) else 0
    content_units = list(lesson.get("content_units") or [])
    content_chunks = chunk_content_units(content_units, size=chunk_size, overlap=chunk_overlap)
    return {
        "metadata_documents": metadata_documents,
        "metadata_documents_skipped_empty": metadata_documents_skipped_empty,
        "content_units": content_units,
        "content_chunks": content_chunks,
        "documents": metadata_documents + content_chunks,
        "skipped_empty_page": not bool(content_units),
        "ocr_units": _count_units(content_units, _is_ocr_unit),
        "vision_images_processed": _count_units(content_units, _is_vision_processed_unit),
        "vision_errors": _count_units(content_units, _has_vision_error),
        "tables_extracted": _count_units(content_units, _is_table_unit),
    }


def _should_index_lesson_metadata_document(lesson: dict[str, Any], metadata_document: Any) -> bool:
    if not metadata_document:
        return False

    metadata = dict(lesson.get("metadata") or {})
    document_metadata = dict(metadata_document.get("metadata") or {}) if isinstance(metadata_document, dict) else {}
    for key, value in document_metadata.items():
        if value not in {"", None}:
            metadata[key] = value
    return should_index_metadata_document(metadata)


def _record_indexed_lesson(
    state: dict[str, Any],
    lesson: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    status: str,
    content_hash: str | None = None,
) -> None:
    page_id = _lesson_page_id(lesson)
    metadata = dict(lesson.get("metadata") or {})
    record = {
        "page_id": page_id,
        "content_hash": content_hash or compute_notion_lesson_hash(lesson),
        "last_edited_time": metadata.get("last_edited_time", ""),
        "title": metadata.get("title", ""),
        "notion_url": metadata.get("notion_url", ""),
        "remote_content_hash": read_remote_content_hash(metadata),
        "indexed_document_ids": [str(document.get("id") or "") for document in documents if document.get("id")],
        "last_synced_at": now_utc_iso(),
        "status": status,
    }
    _state_pages(state)[page_id] = record


def _record_hash_unchanged_lesson(
    state: dict[str, Any],
    lesson: dict[str, Any],
    existing: dict[str, Any],
    content_hash: str,
) -> None:
    page_id = _lesson_page_id(lesson)
    metadata = dict(lesson.get("metadata") or {})
    _state_pages(state)[page_id] = {
        **existing,
        "page_id": page_id,
        "content_hash": content_hash,
        "last_edited_time": metadata.get("last_edited_time", existing.get("last_edited_time", "")),
        "title": metadata.get("title", existing.get("title", "")),
        "notion_url": metadata.get("notion_url", existing.get("notion_url", "")),
        "remote_content_hash": read_remote_content_hash(metadata) or existing.get("remote_content_hash", ""),
        "last_checked_at": now_utc_iso(),
        "status": "hash_unchanged",
    }


def _remove_missing_pages(
    state: dict[str, Any],
    current_page_ids: set[str],
    store: VectorStore,
    summary: dict[str, Any],
) -> None:
    pages = _state_pages(state)
    for page_id in sorted(set(pages) - current_page_ids):
        store.delete_by_page_id(page_id)
        pages.pop(page_id, None)
        summary["pages_removed"] += 1


def _accumulate_document_counts(summary: dict[str, Any], prepared: dict[str, Any]) -> None:
    summary["metadata_documents"] += len(prepared["metadata_documents"])
    summary["metadata_documents_skipped_empty"] += prepared.get("metadata_documents_skipped_empty", 0)
    summary["content_documents"] += len(prepared["content_units"])
    summary["ocr_units"] += prepared.get("ocr_units", 0)
    summary["vision_images_processed"] += prepared.get("vision_images_processed", 0)
    summary["vision_errors"] += prepared.get("vision_errors", 0)
    summary["tables_extracted"] += prepared.get("tables_extracted", 0)
    if prepared["skipped_empty_page"]:
        summary["skipped_empty_pages"] += 1


def _maybe_write_notion_hash(
    lesson: dict[str, Any],
    *,
    state: dict[str, Any],
    summary: dict[str, Any],
    hash_writer_fn: Callable[[str, str], dict[str, Any]],
) -> None:
    if not config.ENABLE_NOTION_HASH_WRITE:
        return
    page_id = _lesson_page_id(lesson)
    content_hash = compute_notion_lesson_hash(lesson)
    try:
        result = hash_writer_fn(page_id, content_hash)
    except Exception as exc:
        summary["hash_write_failed"] += 1
        summary["errors"].append(f"hash write failed for {page_id}: {exc}")
        return

    if result.get("ok") is False:
        summary["hash_write_failed"] += 1
        summary["errors"].append(f"hash write failed for {page_id}: {result.get('error', 'unknown error')}")
        return

    summary["hash_written"] += 1
    record = _state_pages(state).get(page_id)
    if isinstance(record, dict):
        record["remote_content_hash"] = content_hash
        record["hash_written_at"] = now_utc_iso()


def _can_skip_by_remote_hash(existing: Any, metadata: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    remote_content_hash = read_remote_content_hash(metadata)
    if not (remote_content_hash and existing.get("content_hash") and remote_content_hash == existing.get("content_hash")):
        return False

    # Remote hash can only safely skip a changed last_edited_time when that
    # edit was caused by our own previous hash writeback.
    last_edited = _parse_iso_time(metadata.get("last_edited_time"))
    hash_written_at = _parse_iso_time(existing.get("hash_written_at"))
    return bool(last_edited and hash_written_at and last_edited <= hash_written_at)


def _can_skip_by_last_edited(existing: Any, metadata: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    last_edited_time = metadata.get("last_edited_time") or ""
    return bool(
        existing.get("content_hash")
        and existing.get("last_edited_time")
        and last_edited_time
        and existing.get("last_edited_time") == last_edited_time
    )


def _parse_iso_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _lesson_page_id(lesson: dict[str, Any]) -> str:
    metadata = dict(lesson.get("metadata") or {})
    return str(lesson.get("page_id") or metadata.get("page_id") or "")


def _count_units(units: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int:
    return sum(1 for unit in units if predicate(unit))


def _is_ocr_unit(unit: dict[str, Any]) -> bool:
    metadata = dict(unit.get("metadata") or {})
    return bool(metadata.get("ocr_provider") or metadata.get("image_ocr_text"))


def _is_vision_processed_unit(unit: dict[str, Any]) -> bool:
    metadata = dict(unit.get("metadata") or {})
    return bool(metadata.get("ocr_provider"))


def _has_vision_error(unit: dict[str, Any]) -> bool:
    metadata = dict(unit.get("metadata") or {})
    return bool(metadata.get("ocr_error"))


def _is_table_unit(unit: dict[str, Any]) -> bool:
    metadata = dict(unit.get("metadata") or {})
    return metadata.get("block_type") == "table"


def _state_pages(state: dict[str, Any]) -> dict[str, Any]:
    pages = state.setdefault("notion_pages", {})
    if not isinstance(pages, dict):
        state["notion_pages"] = {}
    return state["notion_pages"]


def _empty_summary(sync_mode: str, pages_seen: int = 0) -> dict[str, Any]:
    return {
        "sync_mode": sync_mode,
        "pages_seen": pages_seen,
        "pages_new": 0,
        "pages_changed": 0,
        "pages_skipped": 0,
        "pages_hash_unchanged": 0,
        "pages_removed": 0,
        "metadata_documents": 0,
        "metadata_documents_skipped_empty": 0,
        "content_documents": 0,
        "ocr_units": 0,
        "vision_images_processed": 0,
        "vision_errors": 0,
        "tables_extracted": 0,
        "hash_written": 0,
        "hash_write_failed": 0,
        "chunks_indexed": 0,
        "skipped_empty_pages": 0,
        "errors": [],
        "collection_count": 0,
    }
