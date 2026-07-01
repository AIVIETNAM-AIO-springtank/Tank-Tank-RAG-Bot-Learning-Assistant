"""Tests for full and incremental Notion sync orchestration."""

from __future__ import annotations

from typing import Any

from upgrade_new.src.sync.notion_sync import sync_notion
from upgrade_new.src.sync.sync_state import SyncStateStore


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted_source_type = None
        self.deleted_page_ids: list[str] = []
        self.documents: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []

    def delete_by_source_type(self, source_type: str) -> None:
        self.deleted_source_type = source_type
        self.documents = [doc for doc in self.documents if doc.get("metadata", {}).get("source_type") != source_type]

    def delete_by_page_id(self, page_id: str) -> None:
        self.deleted_page_ids.append(page_id)
        self.documents = [doc for doc in self.documents if doc.get("metadata", {}).get("page_id") != page_id]

    def add_documents(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def count(self) -> int:
        return len(self.documents)


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in texts]


def lesson(page_id: str = "page1", text: str = "Branching content.", edited: str = "t1") -> dict[str, Any]:
    return {
        "page_id": page_id,
        "metadata": {
            "source_type": "notion",
            "page_id": page_id,
            "title": "Basic Python",
            "last_edited_time": edited,
            "notion_url": f"https://notion.so/{page_id}",
        },
        "metadata_document": {
            "id": f"notion_{page_id}_metadata",
            "text": "Bai: Basic Python",
            "metadata": {"source_type": "notion", "source_granularity": "lesson_metadata", "page_id": page_id},
        },
        "content_units": [
            {
                "id": f"{page_id}_block1",
                "text": text,
                "metadata": {
                    "source_type": "notion",
                    "source_granularity": "lesson_content",
                    "page_id": page_id,
                    "title": "Basic Python",
                    "block_id": "block1",
                    "block_type": "paragraph",
                    "heading_path": "Intro",
                },
            }
        ],
    }


def empty_title_lesson(page_id: str = "empty-page", text: str = "Real content from empty title page.") -> dict[str, Any]:
    item = lesson(page_id=page_id, text=text)
    item["metadata"]["title"] = "empty"
    item["metadata_document"]["text"] = "Bai: empty"
    item["metadata_document"]["metadata"]["title"] = "empty"
    item["content_units"][0]["metadata"]["title"] = "empty"
    return item


def state_store(tmp_path) -> SyncStateStore:
    return SyncStateStore(tmp_path / "sync_state.json")


def test_full_sync_rebuilds_notion_source_and_state(tmp_path) -> None:
    store = FakeVectorStore()
    sync_state = state_store(tmp_path)

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        sync_mode="full",
        lessons=[lesson()],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=sync_state,
    )

    assert store.deleted_source_type == "notion"
    assert len(store.documents) == 2
    assert summary["sync_mode"] == "full"
    assert summary["pages_new"] == 1
    assert summary["chunks_indexed"] == 2
    assert sync_state.get_page("page1")["content_hash"]


def test_full_sync_skips_empty_metadata_but_keeps_content_chunks(tmp_path) -> None:
    store = FakeVectorStore()

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        sync_mode="full",
        lessons=[lesson(page_id="valid-page"), empty_title_lesson()],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=state_store(tmp_path),
    )

    metadata_docs = [
        document
        for document in store.documents
        if document.get("metadata", {}).get("source_granularity") == "lesson_metadata"
    ]
    empty_page_docs = [
        document
        for document in store.documents
        if document.get("metadata", {}).get("page_id") == "empty-page"
    ]

    assert len(metadata_docs) == 1
    assert metadata_docs[0]["metadata"].get("page_id") == "valid-page"
    assert len(empty_page_docs) == 1
    assert empty_page_docs[0]["metadata"].get("source_granularity") == "lesson_content"
    assert summary["metadata_documents"] == 1
    assert summary["metadata_documents_skipped_empty"] == 1
    assert summary["content_documents"] == 2
    assert summary["chunks_indexed"] == 3


def test_full_sync_writes_notion_hash_after_successful_index(tmp_path, monkeypatch) -> None:
    store = FakeVectorStore()
    writes: list[tuple[str, str]] = []
    monkeypatch.setattr("upgrade_new.src.config.ENABLE_NOTION_HASH_WRITE", True)

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        sync_mode="full",
        lessons=[lesson(page_id="page-write")],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=state_store(tmp_path),
        hash_writer_fn=lambda page_id, content_hash: writes.append((page_id, content_hash)) or {"ok": True},
    )

    assert writes and writes[0][0] == "page-write"
    assert summary["hash_written"] == 1
    assert summary["hash_write_failed"] == 0


def test_hash_write_failure_does_not_rollback_index(tmp_path, monkeypatch) -> None:
    store = FakeVectorStore()
    monkeypatch.setattr("upgrade_new.src.config.ENABLE_NOTION_HASH_WRITE", True)

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        sync_mode="full",
        lessons=[lesson(page_id="page-fail")],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=state_store(tmp_path),
        hash_writer_fn=lambda page_id, content_hash: {"ok": False, "error": "permission denied"},
    )

    assert len(store.documents) == 2
    assert summary["hash_written"] == 0
    assert summary["hash_write_failed"] == 1
    assert summary["errors"]


def test_incremental_sync_indexes_new_page(tmp_path) -> None:
    store = FakeVectorStore()

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        lessons=[lesson()],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=state_store(tmp_path),
    )

    assert store.deleted_page_ids == ["page1"]
    assert len(store.documents) == 2
    assert len(store.embeddings) == 2
    assert summary["sync_mode"] == "incremental"
    assert summary["pages_new"] == 1
    assert summary["chunks_indexed"] == 2


def test_incremental_sync_skips_when_remote_hash_matches_local_state(tmp_path) -> None:
    sync_state = state_store(tmp_path)
    sync_state.upsert_page(
        "page1",
        {
            "content_hash": "remote-hash",
            "last_edited_time": "old-time",
            "hash_written_at": "2026-06-28T10:05:00+00:00",
            "title": "Basic Python",
            "indexed_document_ids": ["notion_page1_metadata"],
        },
    )
    page = {
        "id": "page1",
        "url": "https://notion.so/page1",
        "last_edited_time": "2026-06-28T10:00:00.000Z",
        "properties": {
            "Bai": {"type": "title", "title": [{"type": "text", "plain_text": "Basic Python", "text": {"content": "Basic Python"}}]},
            "Content Hash": {"type": "rich_text", "rich_text": [{"type": "text", "plain_text": "remote-hash", "text": {"content": "remote-hash"}}]},
        },
    }

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        embed_fn=fake_embed,
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        sync_state_store=sync_state,
        query_pages_fn=lambda: [page],
        page_loader_fn=lambda loaded_page: (_ for _ in ()).throw(AssertionError("should not fetch blocks")),
    )

    assert summary["pages_skipped"] == 1
    assert summary["chunks_indexed"] == 0


def test_incremental_sync_does_not_skip_remote_hash_after_user_edit(tmp_path) -> None:
    sync_state = state_store(tmp_path)
    sync_state.upsert_page(
        "page1",
        {
            "content_hash": "remote-hash",
            "last_edited_time": "old-time",
            "hash_written_at": "2026-06-28T10:05:00+00:00",
            "title": "Basic Python",
            "indexed_document_ids": ["notion_page1_metadata"],
        },
    )
    page = {
        "id": "page1",
        "url": "https://notion.so/page1",
        "last_edited_time": "2026-06-28T10:30:00.000Z",
        "properties": {
            "Bai": {"type": "title", "title": [{"type": "text", "plain_text": "Basic Python", "text": {"content": "Basic Python"}}]},
            "Content Hash": {"type": "rich_text", "rich_text": [{"type": "text", "plain_text": "remote-hash", "text": {"content": "remote-hash"}}]},
        },
    }

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        embed_fn=fake_embed,
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        sync_state_store=sync_state,
        query_pages_fn=lambda: [page],
        page_loader_fn=lambda loaded_page: lesson(page_id="page1", text="User edited content.", edited="2026-06-28T10:30:00.000Z"),
    )

    assert summary["pages_changed"] == 1
    assert summary["chunks_indexed"] == 2


def test_incremental_sync_skips_loaded_lesson_when_hash_unchanged(tmp_path) -> None:
    sync_state = state_store(tmp_path)
    store = FakeVectorStore()
    first = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        lessons=[lesson()],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=sync_state,
    )
    assert first["pages_new"] == 1

    store.documents = []
    store.embeddings = []
    store.deleted_page_ids = []
    second = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        lessons=[lesson()],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=sync_state,
    )

    assert second["pages_hash_unchanged"] == 1
    assert second["chunks_indexed"] == 0
    assert store.deleted_page_ids == []
    assert store.documents == []


def test_incremental_sync_skips_unchanged_live_page_before_fetching_blocks(tmp_path) -> None:
    sync_state = state_store(tmp_path)
    sync_state.upsert_page(
        "page1",
        {
            "content_hash": "abc",
            "last_edited_time": "t1",
            "title": "Basic Python",
            "indexed_document_ids": ["notion_page1_metadata"],
        },
    )

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        embed_fn=fake_embed,
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        sync_state_store=sync_state,
        query_pages_fn=lambda: [{"id": "page1", "url": "https://notion.so/page1", "last_edited_time": "t1", "properties": {}}],
        page_loader_fn=lambda page: (_ for _ in ()).throw(AssertionError("should not fetch blocks")),
    )

    assert summary["pages_skipped"] == 1
    assert summary["chunks_indexed"] == 0


def test_incremental_sync_reindexes_changed_page(tmp_path) -> None:
    sync_state = state_store(tmp_path)
    store = FakeVectorStore()
    sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        lessons=[lesson(text="Old content.", edited="t1")],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=sync_state,
    )

    store.documents = []
    store.embeddings = []
    store.deleted_page_ids = []
    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        lessons=[lesson(text="New content.", edited="t2")],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=sync_state,
    )

    assert summary["pages_changed"] == 1
    assert summary["chunks_indexed"] == 2
    assert store.deleted_page_ids == ["page1"]


def test_incremental_sync_removes_missing_page(tmp_path) -> None:
    sync_state = state_store(tmp_path)
    sync_state.upsert_page("page1", {"content_hash": "abc", "last_edited_time": "t1"})
    store = FakeVectorStore()

    summary = sync_notion(
        chunk_size=200,
        chunk_overlap=20,
        lessons=[],
        embed_fn=fake_embed,
        vector_store=store,  # type: ignore[arg-type]
        sync_state_store=sync_state,
    )

    assert summary["pages_removed"] == 1
    assert store.deleted_page_ids == ["page1"]
    assert sync_state.get_page("page1") is None
