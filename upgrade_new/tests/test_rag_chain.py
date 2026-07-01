"""Tests for RAG chain orchestration."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from upgrade_new.src import rag_chain
from upgrade_new.src.utils.errors import GenerationError


def test_answer_question_passes_retrieval_mode(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_retrieve(question, top_k, filters=None, vector_store=None, retrieval_mode="vector", candidate_k=None):
        captured["question"] = question
        captured["top_k"] = top_k
        captured["retrieval_mode"] = retrieval_mode
        captured["candidate_k"] = candidate_k
        return [{"id": "doc1", "text": "Context text", "metadata": {"source_type": "notion"}}]

    monkeypatch.setattr(rag_chain, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_chain, "generate_with_gemini", lambda prompt: "Answer")

    result = rag_chain.answer_question("What?", top_k=3, retrieval_mode="hybrid_rrf", rerank_enabled=False, mmr_enabled=False)

    assert result["answer"] == "Answer"
    assert captured == {"question": "What?", "top_k": 3, "retrieval_mode": "hybrid_rrf", "candidate_k": 3}


class FakeGeminiResponse:
    def __init__(self, status_code: int, text: str, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def test_generate_with_gemini_rotates_keys_after_quota(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(rag_chain.config, "GEMINI_API_KEYS", ["quota-key", "working-key"])
    monkeypatch.setattr(rag_chain.config, "GEMINI_GENERATION_MODEL", "test-model")

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> FakeGeminiResponse:
        calls.append(url)
        if "quota-key" in url:
            return FakeGeminiResponse(429, '{"error":{"message":"quota"}}')
        return FakeGeminiResponse(
            200,
            "{}",
            {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    assert rag_chain.generate_with_gemini("prompt") == "OK"
    assert len(calls) == 2


def test_generate_with_gemini_masks_keys_in_error(monkeypatch) -> None:
    secret = "secret-key"
    monkeypatch.setattr(rag_chain.config, "GEMINI_API_KEYS", [secret])

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> FakeGeminiResponse:
        raise requests.ConnectionError(f"failed url contains {secret}")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(GenerationError) as exc_info:
        rag_chain.generate_with_gemini("prompt")

    assert secret not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)


def test_generate_with_cohere_parses_chat_response(monkeypatch) -> None:
    monkeypatch.setattr(rag_chain.config, "COHERE_API_KEY", "cohere-key")
    monkeypatch.setattr(rag_chain.config, "COHERE_GENERATION_MODEL", "command-test")

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: int) -> FakeGeminiResponse:
        assert url == "https://api.cohere.com/v2/chat"
        assert json["model"] == "command-test"
        assert headers["Authorization"] == "Bearer cohere-key"
        return FakeGeminiResponse(
            200,
            "{}",
            {"message": {"content": [{"type": "text", "text": "Cohere answer"}]}},
        )

    monkeypatch.setattr(requests, "post", fake_post)

    assert rag_chain.generate_with_cohere("prompt") == "Cohere answer"


def test_generate_answer_auto_falls_back_to_cohere(monkeypatch) -> None:
    monkeypatch.setattr(rag_chain.config, "GENERATION_PROVIDER", "auto")
    monkeypatch.setattr(rag_chain.config, "COHERE_API_KEY", "cohere-key")
    monkeypatch.setattr(rag_chain, "generate_with_gemini", lambda prompt: (_ for _ in ()).throw(GenerationError("429")))
    monkeypatch.setattr(rag_chain, "generate_with_cohere", lambda prompt, previous_error=None: "fallback")

    assert rag_chain.generate_answer("prompt") == "fallback"
