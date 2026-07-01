"""Retrieval orchestration for vector, keyword and RRF hybrid search."""

from __future__ import annotations

from typing import Any, Callable

from upgrade_new.src import config
from upgrade_new.src.embeddings import embed_query
from upgrade_new.src.keyword_store import KeywordStore
from upgrade_new.src.vector_store import VectorStore


VALID_RETRIEVAL_MODES = {"vector", "keyword", "hybrid", "hybrid_rrf"}


def retrieve(
    question: str,
    top_k: int,
    filters: dict[str, Any] | None = None,
    vector_store: VectorStore | None = None,
    embed_fn: Callable[[str], list[float]] = embed_query,
    retrieval_mode: str = "vector",
    candidate_k: int | None = None,
    vector_weight: float | None = None,
    keyword_weight: float | None = None,
    rrf_k: int | None = None,
    keyword_store_factory: Callable[[], KeywordStore] = KeywordStore,
) -> list[dict[str, Any]]:
    """Retrieve structured chunks from ChromaDB using the selected mode."""
    if not question or not question.strip() or top_k <= 0:
        return []
    mode = _normalize_mode(retrieval_mode)
    store = vector_store or VectorStore()
    candidates = candidate_k or config.HYBRID_CANDIDATE_K

    if mode == "vector":
        return _vector_retrieve(question, top_k, filters, store, embed_fn)
    if mode == "keyword":
        return _keyword_retrieve(question, top_k, filters, store, keyword_store_factory)
    return _hybrid_rrf_retrieve(
        question=question,
        top_k=top_k,
        filters=filters,
        store=store,
        embed_fn=embed_fn,
        keyword_store_factory=keyword_store_factory,
        candidate_k=max(top_k, candidates),
        rrf_k=config.RRF_K if rrf_k is None else rrf_k,
    )


def _vector_retrieve(
    question: str,
    top_k: int,
    filters: dict[str, Any] | None,
    store: VectorStore,
    embed_fn: Callable[[str], list[float]],
) -> list[dict[str, Any]]:
    query_embedding = embed_fn(question)
    results = store.query(query_embedding=query_embedding, top_k=top_k, where=filters)
    scored = _attach_vector_scores(results)
    for item in scored:
        item["retrieval_mode"] = "vector"
    return scored


def _keyword_retrieve(
    question: str,
    top_k: int,
    filters: dict[str, Any] | None,
    store: VectorStore,
    keyword_store_factory: Callable[[], KeywordStore],
) -> list[dict[str, Any]]:
    documents = store.get_documents(where=filters)
    keyword_store = keyword_store_factory()
    keyword_store.build(documents)
    return keyword_store.query(question, top_k=top_k)


def _hybrid_rrf_retrieve(
    *,
    question: str,
    top_k: int,
    filters: dict[str, Any] | None,
    store: VectorStore,
    embed_fn: Callable[[str], list[float]],
    keyword_store_factory: Callable[[], KeywordStore],
    candidate_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    vector_results = _vector_retrieve(question, candidate_k, filters, store, embed_fn)
    keyword_results = _keyword_retrieve(question, candidate_k, filters, store, keyword_store_factory)

    vector_ranks = _rank_map(vector_results)
    keyword_ranks = _rank_map(keyword_results)

    merged: dict[str, dict[str, Any]] = {}
    for item in vector_results + keyword_results:
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        existing = merged.setdefault(item_id, dict(item))
        existing.setdefault("id", item_id)
        existing.setdefault("text", item.get("text", ""))
        existing.setdefault("metadata", item.get("metadata", {}))
        if item.get("distance") is not None:
            existing["distance"] = item.get("distance")
        if item.get("vector_score") is not None:
            existing["vector_score"] = item.get("vector_score")
        if item.get("keyword_score") is not None:
            existing["keyword_score"] = item.get("keyword_score")

    for item_id, item in merged.items():
        vector_rank = vector_ranks.get(item_id)
        keyword_rank = keyword_ranks.get(item_id)
        item["vector_rank"] = vector_rank
        item["keyword_rank"] = keyword_rank
        item["rrf_score"] = _rrf_score(vector_rank, keyword_rank, rrf_k=rrf_k)
        item["retrieval_mode"] = "hybrid_rrf"

    return sorted(
        merged.values(),
        key=lambda item: (
            item.get("rrf_score", 0.0),
            -(item.get("vector_rank") or candidate_k + 1),
            -(item.get("keyword_rank") or candidate_k + 1),
        ),
        reverse=True,
    )[:top_k]


def _attach_vector_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for item in results:
        copy = dict(item)
        distance = copy.get("distance")
        copy["vector_score"] = _distance_to_score(distance)
        scored.append(copy)
    return scored


def _distance_to_score(distance: Any) -> float:
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if value < 0:
        value = 0.0
    return 1.0 / (1.0 + value)


def _rank_map(results: list[dict[str, Any]]) -> dict[str, int]:
    """Return 1-based ranks for retrieval results, preserving first occurrence."""
    ranks: dict[str, int] = {}
    for rank, item in enumerate(results, start=1):
        item_id = str(item.get("id") or "")
        if item_id and item_id not in ranks:
            ranks[item_id] = rank
    return ranks


def _rrf_score(vector_rank: int | None, keyword_rank: int | None, *, rrf_k: int) -> float:
    """Compute Reciprocal Rank Fusion score from optional channel ranks."""
    rank_constant = max(1, int(rrf_k))
    score = 0.0
    if vector_rank is not None:
        score += 1.0 / (rank_constant + vector_rank)
    if keyword_rank is not None:
        score += 1.0 / (rank_constant + keyword_rank)
    return score


def _normalize_mode(mode: str) -> str:
    clean = (mode or "vector").strip().lower()
    if clean == "hybrid":
        return "hybrid_rrf"
    if clean not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of {sorted(VALID_RETRIEVAL_MODES)}")
    return clean
