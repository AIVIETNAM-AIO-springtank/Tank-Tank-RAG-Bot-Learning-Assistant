"""Tests for persistent ChromaDB vector store behavior."""

from __future__ import annotations

from upgrade_new.src.vector_store import VectorStore


def test_vector_store_add_query_and_delete(tmp_path) -> None:
    store = VectorStore(path=str(tmp_path / "chroma"), collection_name="test_collection")
    documents = [
        {
            "id": "doc1_chunk_0",
            "text": "alpha beta content",
            "metadata": {
                "source_type": "pdf",
                "source_file": "lesson.pdf",
                "page_number": 1,
                "document_id": "doc1",
                "chunk_id": 0,
            },
        }
    ]

    store.add_documents(documents, embeddings=[[1.0, 0.0, 0.0]])
    results = store.query([1.0, 0.0, 0.0], top_k=1)
    stored_documents = store.get_documents(where={"document_id": "doc1"})

    assert len(results) == 1
    assert results[0]["id"] == "doc1_chunk_0"
    assert results[0]["text"] == "alpha beta content"
    assert results[0]["metadata"]["document_id"] == "doc1"
    assert results[0]["distance"] is not None
    assert stored_documents[0]["id"] == "doc1_chunk_0"
    assert stored_documents[0]["metadata"]["source_file"] == "lesson.pdf"

    store.delete_by_document_id("doc1")
    assert store.count() == 0


def test_vector_store_delete_by_source_type_and_page_id(tmp_path) -> None:
    store = VectorStore(path=str(tmp_path / "chroma"), collection_name="test_collection_delete")
    documents = [
        {
            "id": "notion_page1_chunk_0",
            "text": "notion content one",
            "metadata": {"source_type": "notion", "page_id": "page1", "chunk_id": 0},
        },
        {
            "id": "notion_page2_chunk_0",
            "text": "notion content two",
            "metadata": {"source_type": "notion", "page_id": "page2", "chunk_id": 0},
        },
    ]
    store.add_documents(documents, embeddings=[[1.0, 0.0], [0.0, 1.0]])

    store.delete_by_page_id("page1")
    assert store.count() == 1
    store.delete_by_source_type("notion")
    assert store.count() == 0
