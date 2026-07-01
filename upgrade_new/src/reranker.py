"""Optional reranking and final context selection for RAG."""

from __future__ import annotations

import re
from typing import Any, Callable

import requests

from upgrade_new.src import config
from upgrade_new.src.keyword_store import tokenize


COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


def rerank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    provider: str | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    request_fn: Callable[..., requests.Response] | None = None,
) -> list[dict[str, Any]]:
    """Rerank candidates with an optional provider and safe fallback."""
    if top_k <= 0 or not candidates:
        return []

    selected_provider = (provider if provider is not None else config.RERANK_PROVIDER).strip().lower()
    if selected_provider in {"", "none", "off", "disabled", "false"}:
        return _copy_top(candidates, top_k)

    if selected_provider != "cohere":
        return _copy_top(candidates, top_k, rerank_error=f"Unsupported rerank provider: {selected_provider}")

    try:
        return _rerank_with_cohere(
            question=question,
            candidates=candidates,
            top_k=top_k,
            model=model or config.COHERE_RERANK_MODEL,
            api_key=api_key or config.COHERE_API_KEY,
            request_fn=request_fn or requests.post,
        )
    except Exception as exc:
        return _copy_top(candidates, top_k, rerank_error=f"Cohere rerank fallback: {exc}")


def select_context_mmr(
    question: str,
    candidates: list[dict[str, Any]],
    final_k: int,
    lambda_mult: float = config.MMR_LAMBDA,
    min_text_similarity: float = config.MMR_MIN_TEXT_SIMILARITY,
) -> list[dict[str, Any]]:
    """Select diverse final context chunks using a lightweight MMR pass."""
    if final_k <= 0 or not candidates:
        return []

    unique_candidates = _dedupe_candidates(candidates)
    if len(unique_candidates) <= final_k:
        return [_mark_mmr(item, rank=index + 1, score=1.0, penalty=0.0) for index, item in enumerate(unique_candidates)]

    relevances = _normalized_relevance(unique_candidates, question)
    selected: list[dict[str, Any]] = []
    remaining = list(range(len(unique_candidates)))
    lambda_value = min(max(lambda_mult, 0.0), 1.0)
    similarity_limit = min(max(min_text_similarity, 0.0), 1.0)

    while remaining and len(selected) < final_k:
        scored: list[tuple[float, bool, int, float]] = []
        for candidate_index in remaining:
            candidate = unique_candidates[candidate_index]
            penalty = _max_similarity(candidate, selected)
            relevance = relevances[candidate_index]
            score = lambda_value * relevance - (1.0 - lambda_value) * penalty
            is_duplicate = bool(selected and penalty >= similarity_limit)
            scored.append((score, is_duplicate, candidate_index, penalty))

        non_duplicates = [item for item in scored if not item[1]]
        pool = non_duplicates or scored
        score, is_duplicate, chosen_index, penalty = max(pool, key=lambda item: (item[0], relevances[item[2]]))
        chosen = _mark_mmr(
            unique_candidates[chosen_index],
            rank=len(selected) + 1,
            score=score,
            penalty=penalty,
            duplicate_allowed=is_duplicate,
        )
        selected.append(chosen)
        remaining.remove(chosen_index)

    return selected


def _rerank_with_cohere(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    model: str,
    api_key: str,
    request_fn: Callable[..., requests.Response],
) -> list[dict[str, Any]]:
    if not api_key:
        raise RuntimeError("missing COHERE_API_KEY")

    documents = [_format_rerank_document(item) for item in candidates]
    payload = {
        "model": model,
        "query": question,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = request_fn(COHERE_RERANK_URL, json=payload, headers=headers, timeout=config.REQUEST_TIMEOUT)
    if response.status_code >= 400:
        detail = response.text[:240].replace("\n", " ")
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")

    data = response.json()
    ranked: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for rank, result in enumerate(data.get("results", []), start=1):
        original_index = int(result.get("index", -1))
        if original_index < 0 or original_index >= len(candidates) or original_index in seen_indexes:
            continue
        seen_indexes.add(original_index)
        item = dict(candidates[original_index])
        item["rerank_provider"] = "cohere"
        item["rerank_model"] = model
        item["rerank_rank"] = rank
        item["rerank_score"] = float(result.get("relevance_score", 0.0))
        ranked.append(item)

    return ranked or _copy_top(candidates, top_k, rerank_error="Cohere returned no rerank results")


def _format_rerank_document(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    metadata_bits = [
        f"title: {metadata.get('title', '')}",
        f"source_type: {metadata.get('source_type', '')}",
        f"heading_path: {metadata.get('heading_path', '')}",
        f"page_number: {metadata.get('page_number', '')}",
    ]
    text = str(candidate.get("text") or "")
    text = re.sub(r"\s+", " ", text).strip()
    return "\n".join(bit for bit in metadata_bits if not bit.endswith(": ")) + f"\ntext: {text[:6000]}"


def _copy_top(candidates: list[dict[str, Any]], top_k: int, rerank_error: str | None = None) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:top_k], start=1):
        item = dict(candidate)
        item.setdefault("rerank_rank", index)
        item.setdefault("rerank_provider", "none")
        if rerank_error:
            item["rerank_error"] = rerank_error
        copied.append(item)
    return copied


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for candidate in candidates:
        item_id = str(candidate.get("id") or "")
        fingerprint = _text_fingerprint(str(candidate.get("text") or ""))
        if item_id and item_id in seen_ids:
            continue
        if fingerprint and fingerprint in seen_fingerprints:
            continue
        if item_id:
            seen_ids.add(item_id)
        if fingerprint:
            seen_fingerprints.add(fingerprint)
        unique.append(dict(candidate))
    return unique


def _normalized_relevance(candidates: list[dict[str, Any]], question: str) -> list[float]:
    raw_scores = [_raw_relevance(candidate, index=index, question=question) for index, candidate in enumerate(candidates)]
    minimum = min(raw_scores)
    maximum = max(raw_scores)
    if maximum <= minimum:
        return [1.0 - (index / max(1, len(candidates))) for index in range(len(candidates))]
    return [(score - minimum) / (maximum - minimum) for score in raw_scores]


def _raw_relevance(candidate: dict[str, Any], *, index: int, question: str) -> float:
    for key in ("rerank_score", "rrf_score", "hybrid_score", "vector_score", "keyword_score"):
        value = candidate.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return _token_overlap(question, str(candidate.get("text") or "")) + 1.0 / (index + 1)


def _max_similarity(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    candidate_text = str(candidate.get("text") or "")
    return max(_text_similarity(candidate_text, str(item.get("text") or "")) for item in selected)


def _text_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _token_overlap(question: str, text: str) -> float:
    question_tokens = set(tokenize(question))
    text_tokens = set(tokenize(text))
    if not question_tokens or not text_tokens:
        return 0.0
    return len(question_tokens & text_tokens) / len(question_tokens)


def _text_fingerprint(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    return compact[:500]


def _mark_mmr(
    candidate: dict[str, Any],
    *,
    rank: int,
    score: float,
    penalty: float,
    duplicate_allowed: bool = False,
) -> dict[str, Any]:
    item = dict(candidate)
    item["mmr_selected"] = True
    item["mmr_rank"] = rank
    item["mmr_score"] = score
    item["mmr_diversity_penalty"] = penalty
    if duplicate_allowed:
        item["mmr_duplicate_allowed"] = True
    return item
