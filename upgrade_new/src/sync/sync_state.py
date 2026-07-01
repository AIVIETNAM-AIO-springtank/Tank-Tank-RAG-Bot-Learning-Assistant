"""Local JSON sync state for Notion incremental sync."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upgrade_new.src import config


STATE_VERSION = 1


class SyncStateStore:
    """Read and write local sync metadata used by incremental Notion sync."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else config.UPGRADE_ROOT / "sync_state.json"

    def load(self) -> dict[str, Any]:
        """Load state from disk, returning an empty state when missing/invalid."""
        if not self.path.exists():
            return self.empty_state()
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return self.empty_state()
        if not isinstance(data, dict):
            return self.empty_state()
        data.setdefault("version", STATE_VERSION)
        data.setdefault("notion_pages", {})
        if not isinstance(data["notion_pages"], dict):
            data["notion_pages"] = {}
        return data

    def save(self, state: dict[str, Any]) -> None:
        """Persist state atomically enough for local development use."""
        clean_state = deepcopy(state)
        clean_state.setdefault("version", STATE_VERSION)
        clean_state.setdefault("notion_pages", {})
        clean_state["updated_at"] = now_utc_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(clean_state, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        temp_path.replace(self.path)

    def get_page(self, page_id: str) -> dict[str, Any] | None:
        """Return sync record for one Notion page, if present."""
        if not page_id:
            return None
        state = self.load()
        record = state.get("notion_pages", {}).get(page_id)
        return deepcopy(record) if isinstance(record, dict) else None

    def upsert_page(self, page_id: str, record: dict[str, Any]) -> None:
        """Insert or replace one Notion page sync record."""
        if not page_id:
            return
        state = self.load()
        pages = state.setdefault("notion_pages", {})
        pages[page_id] = {**record, "page_id": page_id}
        self.save(state)

    def remove_page(self, page_id: str) -> None:
        """Remove one Notion page sync record."""
        if not page_id:
            return
        state = self.load()
        state.setdefault("notion_pages", {}).pop(page_id, None)
        self.save(state)

    def clear_notion_pages(self) -> None:
        """Clear all Notion page records while keeping the state file valid."""
        state = self.load()
        state["notion_pages"] = {}
        self.save(state)

    @staticmethod
    def empty_state() -> dict[str, Any]:
        """Return a new empty sync state."""
        return {"version": STATE_VERSION, "notion_pages": {}}


def now_utc_iso() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
