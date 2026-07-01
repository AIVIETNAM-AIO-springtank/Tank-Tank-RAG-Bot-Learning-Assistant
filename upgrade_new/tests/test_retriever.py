"""Tests for retrieval orchestration without live Cohere calls."""

from __future__ import annotations

from typing import Any

from upgrade_new.src.retriever import retrieve


class FakeVectorStore:
    def __init__(self) -> None:
        self.query_embedding = None
        self.top_k = None
        self.where = None
        self.documents = [
            {
                "id": "doc1_chunk_0",
                "text": "retrieved text",
                "metadata": {"source_type": "pdf", "document_id": "doc1"},
                "distance": None,
            },
            {
                "id": "doc2_chunk_0",
                "text": "Python match-case branching content",
                "metadata": {"source_type": "notion", "page_id": "doc2"},
                "distance": None,
            },
        ]

    def query(self, query_embedding: list[float], top_k: int, where: dict[str, Any] | None = None) -> list[dict]:
        self.query_embedding = query_embedding
        self.top_k = top_k
        self.where = where
        return [
            {
                "id": "doc1_chunk_0",
                "text": "retrieved text",
                "metadata": {"source_type": "pdf", "document_id": "doc1"},
                "distance": 0.12,
            }
        ]

    def get_documents(self, where: dict[str, Any] | None = None) -> list[dict]:
        if not where:
            return self.documents
        return [
            document
            for document in self.documents
            if all(document.get("metadata", {}).get(key) == value for key, value in where.items())
        ]


def test_retrieve_returns_structured_results_with_metadata() -> None:
    store = FakeVectorStore()

    results = retrieve(
        "What is inside?",
        top_k=3,
        filters={"source_type": "pdf"},
        vector_store=store,  # type: ignore[arg-type]
        embed_fn=lambda question: [0.1, 0.2, 0.3],
    )

    assert store.query_embedding == [0.1, 0.2, 0.3]
    assert store.top_k == 3
    assert store.where == {"source_type": "pdf"}
    assert results[0]["text"] == "retrieved text"
    assert results[0]["metadata"]["document_id"] == "doc1"
    assert results[0]["distance"] == 0.12
    assert results[0]["retrieval_mode"] == "vector"


def test_retrieve_keyword_mode_does_not_embed_query() -> None:
    store = FakeVectorStore()

    results = retrieve(
        "match-case",
        top_k=2,
        vector_store=store,  # type: ignore[arg-type]
        retrieval_mode="keyword",
        embed_fn=lambda question: (_ for _ in ()).throw(AssertionError("should not embed keyword-only query")),
    )

    assert results[0]["id"] == "doc2_chunk_0"
    assert results[0]["keyword_score"] > 0
    assert results[0]["retrieval_mode"] == "keyword"


def test_retrieve_hybrid_mode_merges_vector_and_keyword_results() -> None:
    store = FakeVectorStore()

    results = retrieve(
        "match-case",
        top_k=2,
        vector_store=store,  # type: ignore[arg-type]
        retrieval_mode="hybrid_rrf",
        embed_fn=lambda question: [0.1, 0.2, 0.3],
        candidate_k=3,
    )

    ids = [result["id"] for result in results]
    assert "doc1_chunk_0" in ids
    assert "doc2_chunk_0" in ids
    assert all(result["retrieval_mode"] == "hybrid_rrf" for result in results)
    assert all("rrf_score" in result for result in results)
    assert any(result.get("vector_rank") == 1 for result in results)
    assert any(result.get("keyword_rank") == 1 for result in results)


def test_retrieve_hybrid_alias_uses_rrf() -> None:
    store = FakeVectorStore()

    results = retrieve(
        "match-case",
        top_k=2,
        vector_store=store,  # type: ignore[arg-type]
        retrieval_mode="hybrid",
        embed_fn=lambda question: [0.1, 0.2, 0.3],
        candidate_k=3,
    )

    assert results[0]["retrieval_mode"] == "hybrid_rrf"


def test_rrf_promotes_document_found_by_both_channels() -> None:
    class FixedVectorStore(FakeVectorStore):
        def query(self, query_embedding: list[float], top_k: int, where: dict[str, Any] | None = None) -> list[dict]:
            return [
                {"id": "vector_only", "text": "semantic only", "metadata": {}, "distance": 0.01},
                {"id": "shared", "text": "semantic and keyword", "metadata": {}, "distance": 0.02},
            ]

    class FixedKeywordStore:
        def build(self, documents: list[dict[str, Any]]) -> None:
            return None

        def query(self, question: str, top_k: int) -> list[dict[str, Any]]:
            return [
                {"id": "keyword_only", "text": "keyword only", "metadata": {}, "keyword_score": 9.0},
                {"id": "shared", "text": "semantic and keyword", "metadata": {}, "keyword_score": 8.0},
            ]

    results = retrieve(
        "shared",
        top_k=3,
        vector_store=FixedVectorStore(),  # type: ignore[arg-type]
        retrieval_mode="hybrid_rrf",
        embed_fn=lambda question: [0.1],
        candidate_k=3,
        rrf_k=60,
        keyword_store_factory=FixedKeywordStore,  # type: ignore[arg-type]
    )

    assert results[0]["id"] == "shared"
    assert results[0]["vector_rank"] == 2
    assert results[0]["keyword_rank"] == 2
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
