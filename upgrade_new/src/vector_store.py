"""Persistent ChromaDB vector store wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from upgrade_new.src import config
from upgrade_new.src.utils.errors import VectorStoreError


class VectorStore:
    """Small wrapper around ChromaDB PersistentClient."""

    def __init__(self, path: str | None = None, collection_name: str | None = None) -> None:
        self.path = path or config.CHROMA_PATH
        self.collection_name = collection_name or config.COLLECTION_NAME
        Path(self.path).mkdir(parents=True, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(path=self.path)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:
            raise VectorStoreError("Không khởi tạo được ChromaDB persistent store.") from exc

    def add_documents(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        """Upsert canonical documents into ChromaDB."""
        if not documents:
            return
        if len(documents) != len(embeddings):
            raise VectorStoreError("Số document và số embedding không khớp.")

        ids = [str(document["id"]) for document in documents]
        texts = [str(document["text"]) for document in documents]
        metadatas = [_sanitize_metadata(document.get("metadata", {})) for document in documents]

        try:
            self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        except Exception as exc:
            raise VectorStoreError("Không lưu được documents vào ChromaDB.") from exc

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query top-k chunks and return structured retrieval results."""
        if not query_embedding:
            return []
        if top_k <= 0:
            return []

        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Không query được ChromaDB.") from exc

        ids = _first(result.get("ids"))
        documents = _first(result.get("documents"))
        metadatas = _first(result.get("metadatas"))
        distances = _first(result.get("distances"))

        items: list[dict[str, Any]] = []
        for index, text in enumerate(documents):
            items.append(
                {
                    "id": ids[index] if index < len(ids) else "",
                    "text": text,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )
        return items

    def get_documents(self, where: dict[str, Any] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Return canonical documents currently stored in ChromaDB."""
        try:
            kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
            if where:
                kwargs["where"] = where
            if limit is not None:
                kwargs["limit"] = limit
            result = self.collection.get(**kwargs)
        except Exception as exc:
            raise VectorStoreError("Could not read documents from ChromaDB.") from exc

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        items: list[dict[str, Any]] = []
        for index, text in enumerate(documents):
            items.append(
                {
                    "id": ids[index] if index < len(ids) else "",
                    "text": text,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": None,
                }
            )
        return items

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete chunks belonging to one source document."""
        if not document_id:
            return
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError("Không xóa được chunks theo document_id.") from exc

    def delete_by_source_type(self, source_type: str) -> None:
        """Delete chunks belonging to one source type, such as pdf or notion."""
        if not source_type:
            return
        try:
            self.collection.delete(where={"source_type": source_type})
        except Exception as exc:
            raise VectorStoreError("Could not delete chunks by source_type.") from exc

    def delete_by_page_id(self, page_id: str) -> None:
        """Delete chunks belonging to one Notion page."""
        if not page_id:
            return
        try:
            self.collection.delete(where={"page_id": page_id})
        except Exception as exc:
            raise VectorStoreError("Could not delete chunks by page_id.") from exc

    def count(self) -> int:
        """Return collection item count."""
        return int(self.collection.count())


def _first(value: Any) -> list[Any]:
    if isinstance(value, list) and value:
        first = value[0]
        return first if isinstance(first, list) else value
    return []


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, tuple, set)):
            clean[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            clean[key] = json.dumps(value, ensure_ascii=False)
        else:
            clean[key] = str(value)
    return clean
