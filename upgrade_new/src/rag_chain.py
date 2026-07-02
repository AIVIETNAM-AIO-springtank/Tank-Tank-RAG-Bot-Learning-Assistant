"""RAG answer generation with Gemini and source citations."""

from __future__ import annotations

from typing import Any

import requests

from upgrade_new.src import config
from upgrade_new.src.prompts import RAG_ANSWER_PROMPT
from upgrade_new.src.reranker import rerank_candidates, select_context_mmr
from upgrade_new.src.retriever import retrieve
from upgrade_new.src.utils.errors import ConfigError, GenerationError
from upgrade_new.src.vector_store import VectorStore


def answer_question(
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
    vector_store: VectorStore | None = None,
    retrieval_mode: str = "vector",
    candidate_k: int | None = None,
    rerank_enabled: bool | None = None,
    rerank_provider: str | None = None,
    rerank_top_n: int | None = None,
    mmr_enabled: bool | None = None,
    mmr_lambda: float | None = None,
    final_context_k: int | None = None,
) -> dict[str, Any]:
    """Answer one question and return answer text plus retrieved sources."""
    final_k = max(1, int(final_context_k or top_k or config.FINAL_CONTEXT_K))
    should_rerank = config.ENABLE_RERANKING if rerank_enabled is None else rerank_enabled
    should_mmr = config.ENABLE_MMR if mmr_enabled is None else mmr_enabled
    retrieve_k = max(
        final_k,
        int(candidate_k or (config.RERANK_CANDIDATE_K if should_rerank or should_mmr else top_k)),
    )

    candidates = retrieve(
        question,
        top_k=retrieve_k,
        filters=filters,
        vector_store=vector_store,
        retrieval_mode=retrieval_mode,
        candidate_k=retrieve_k,
    )
    if not candidates:
        return {
            "answer": "Toi chua tim thay thong tin nay trong tai lieu da index.",
            "sources": [],
            "retrieval_debug": {"retrieved_count": 0, "reranked_count": 0, "final_count": 0},
        }

    reranked = candidates
    if should_rerank:
        reranked = rerank_candidates(
            question,
            candidates,
            top_k=max(final_k, int(rerank_top_n or config.RERANK_TOP_N)),
            provider=rerank_provider,
        )

    sources = (
        select_context_mmr(
            question,
            reranked,
            final_k=final_k,
            lambda_mult=config.MMR_LAMBDA if mmr_lambda is None else mmr_lambda,
        )
        if should_mmr
        else reranked[:final_k]
    )
    if not sources:
        return {
            "answer": "Toi chua tim thay thong tin nay trong tai lieu da index.",
            "sources": [],
            "retrieval_debug": {
                "retrieved_count": len(candidates),
                "reranked_count": len(reranked),
                "final_count": 0,
            },
        }

    context = build_context(sources)
    prompt = RAG_ANSWER_PROMPT.format(context=context, question=question)
    answer = generate_answer(prompt)
    return {
        "answer": answer,
        "sources": sources,
        "retrieval_debug": {
            "retrieved_count": len(candidates),
            "reranked_count": len(reranked),
            "final_count": len(sources),
            "rerank_enabled": should_rerank,
            "mmr_enabled": should_mmr,
        },
    }


def build_context(sources: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the RAG prompt."""
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata", {})
        label = _source_label(metadata)
        blocks.append(f"[{index}] {label}\n{source.get('text', '')}")
    return "\n\n".join(blocks)


def generate_answer(prompt: str, provider: str | None = None) -> str:
    """Generate an answer using the configured provider."""
    selected_provider = (provider or config.GENERATION_PROVIDER or "gemini").strip().lower()
    if selected_provider == "cohere":
        return generate_with_cohere(prompt)
    if selected_provider == "auto":
        try:
            return generate_with_gemini(prompt)
        except Exception as exc:
            if not config.COHERE_API_KEY:
                raise
            return generate_with_cohere(prompt, previous_error=exc)
    return generate_with_gemini(prompt)


def generate_with_gemini(prompt: str) -> str:
    """Call Gemini generateContent with a completed prompt."""
    api_keys = list(config.GEMINI_API_KEYS)
    if not api_keys:
        raise ConfigError("Missing GEMINI_API_KEY or GEMINI_API_KEYS. Add it to environment, .env or Streamlit secrets.")

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }

    errors: list[str] = []
    for index, api_key in enumerate(api_keys, start=1):
        url = _gemini_generate_url(api_key)
        try:
            response = requests.post(url, json=payload, timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            errors.append(f"key {index}/{len(api_keys)} request error: {_safe_gemini_error_detail(str(exc))}")
            continue

        if response.status_code >= 400:
            detail = _safe_gemini_error_detail(response.text)
            errors.append(f"key {index}/{len(api_keys)} Gemini API error {response.status_code}: {detail}")
            continue

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "Toi chua nhan duoc cau tra loi tu Gemini."
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()

    raise GenerationError(f"Gemini API failed for {len(api_keys)} configured key(s): {' | '.join(errors)}")


def generate_with_cohere(prompt: str, previous_error: Exception | None = None) -> str:
    """Call Cohere Chat v2 for RAG answer generation."""
    if not config.COHERE_API_KEY:
        raise ConfigError("Missing COHERE_API_KEY. Add it to environment, .env or Streamlit secrets.")

    payload = {
        "model": config.COHERE_GENERATION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 768,
    }
    headers = {
        "Authorization": f"Bearer {config.COHERE_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post("https://api.cohere.com/v2/chat", json=payload, headers=headers, timeout=config.REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        detail = _safe_cohere_error_detail(str(exc))
        if previous_error is not None:
            detail = f"Gemini fallback failed first: {_safe_gemini_error_detail(str(previous_error))}; Cohere request error: {detail}"
        raise GenerationError(f"Could not call Cohere Chat API. {detail}") from exc

    if response.status_code >= 400:
        detail = _safe_cohere_error_detail(response.text)
        if previous_error is not None:
            detail = f"Gemini fallback failed first: {_safe_gemini_error_detail(str(previous_error))}; Cohere error: {detail}"
        raise GenerationError(f"Cohere Chat API error {response.status_code}: {detail}")

    data = response.json()
    content = data.get("message", {}).get("content", [])
    if isinstance(content, list):
        text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    else:
        text = str(content or "")
    return text.strip() or "Toi chua nhan duoc cau tra loi tu Cohere."


def _gemini_generate_url(api_key: str) -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_GENERATION_MODEL}:generateContent?key={api_key}"
    )


def _safe_gemini_error_detail(detail: str) -> str:
    clean = detail.replace("\n", " ").strip()
    for secret in list(config.GEMINI_API_KEYS) + [config.GEMINI_API_KEY]:
        if secret:
            clean = clean.replace(secret, "<redacted>")
    return clean[:500]


def _safe_cohere_error_detail(detail: str) -> str:
    clean = detail.replace("\n", " ").strip()
    if config.COHERE_API_KEY:
        clean = clean.replace(config.COHERE_API_KEY, "<redacted>")
    return clean[:500]


def _source_label(metadata: dict[str, Any]) -> str:
    source_type = metadata.get("source_type", "unknown")
    if source_type == "pdf":
        return (
            f"PDF: {metadata.get('source_file', '')}, "
            f"page {metadata.get('page_number', '')}, "
            f"chunk {metadata.get('chunk_id', '')}"
        )
    if source_type == "notion":
        return f"Notion: {metadata.get('title', '')}, chunk {metadata.get('chunk_id', '')}"
    return f"Source: {source_type}"
