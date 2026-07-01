"""Cohere embedding wrapper for documents and queries."""

from __future__ import annotations

from typing import Any

import requests

from upgrade_new.src import config
from upgrade_new.src.utils.errors import ConfigError, EmbeddingError

COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document texts with Cohere."""
    return _embed_texts(texts, input_type="search_document")


def embed_query(text: str) -> list[float]:
    """Embed a retrieval query with Cohere."""
    embeddings = _embed_texts([text], input_type="search_query")
    return embeddings[0] if embeddings else []


def _embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    clean_texts = [text for text in texts if text and text.strip()]
    if not clean_texts:
        return []

    if not config.COHERE_API_KEY:
        raise ConfigError("Missing COHERE_API_KEY. Add it to environment, .env or Streamlit secrets.")

    all_embeddings: list[list[float]] = []
    for start in range(0, len(clean_texts), config.BATCH_SIZE):
        batch = clean_texts[start : start + config.BATCH_SIZE]
        all_embeddings.extend(_embed_batch(batch, input_type=input_type))
    return all_embeddings


def _embed_batch(texts: list[str], input_type: str) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {config.COHERE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "texts": texts,
        "model": config.COHERE_EMBEDDING_MODEL,
        "input_type": input_type,
        "embedding_types": ["float"],
    }

    try:
        response = requests.post(COHERE_EMBED_URL, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise EmbeddingError("Không gọi được Cohere embedding API. Hãy kiểm tra mạng và API key.") from exc

    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        raise EmbeddingError(f"Cohere embedding API lỗi {response.status_code}: {detail}")

    data = response.json()
    return _parse_embeddings(data)


def _parse_embeddings(data: dict[str, Any]) -> list[list[float]]:
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list):
        return embeddings
    if isinstance(embeddings, dict):
        float_embeddings = embeddings.get("float")
        if isinstance(float_embeddings, list):
            return float_embeddings
        for value in embeddings.values():
            if isinstance(value, list):
                return value
    raise EmbeddingError("Cohere response không chứa embeddings hợp lệ.")

