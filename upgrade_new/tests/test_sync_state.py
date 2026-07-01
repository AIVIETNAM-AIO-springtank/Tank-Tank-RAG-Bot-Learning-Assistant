"""Tests for local sync state persistence."""

from __future__ import annotations

from upgrade_new.src.sync.sync_state import SyncStateStore


def test_sync_state_loads_empty_when_missing(tmp_path) -> None:
    store = SyncStateStore(tmp_path / "missing.json")

    state = store.load()

    assert state["version"] == 1
    assert state["notion_pages"] == {}


def test_sync_state_save_load_roundtrip(tmp_path) -> None:
    store = SyncStateStore(tmp_path / "sync_state.json")
    state = store.empty_state()
    state["notion_pages"]["page1"] = {"content_hash": "abc", "last_edited_time": "t1"}

    store.save(state)
    loaded = store.load()

    assert loaded["notion_pages"]["page1"]["content_hash"] == "abc"
    assert loaded["updated_at"]


def test_sync_state_page_helpers(tmp_path) -> None:
    store = SyncStateStore(tmp_path / "sync_state.json")

    store.upsert_page("page1", {"content_hash": "abc"})
    assert store.get_page("page1")["content_hash"] == "abc"

    store.remove_page("page1")
    assert store.get_page("page1") is None

    store.upsert_page("page2", {"content_hash": "def"})
    store.clear_notion_pages()
    assert store.load()["notion_pages"] == {}
