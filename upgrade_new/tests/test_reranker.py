"""Tests for optional reranking and MMR context selection."""

from __future__ import annotations

from typing import Any

from upgrade_new.src.reranker import rerank_candidates, select_context_mmr


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


def test_cohere_rerank_uses_mocked_response_order() -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: int) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["auth"] = headers["Authorization"]
        captured["timeout"] = timeout
        return FakeResponse(
            200,
            {
                "results": [
                    {"index": 1, "relevance_score": 0.98},
                    {"index": 0, "relevance_score": 0.42},
                ]
            },
        )

    candidates = [
        {"id": "a", "text": "less relevant", "metadata": {}},
        {"id": "b", "text": "more relevant", "metadata": {}},
    ]

    results = rerank_candidates(
        "question",
        candidates,
        top_k=2,
        provider="cohere",
        model="rerank-v3.5",
        api_key="test-key",
        request_fn=fake_post,  # type: ignore[arg-type]
    )

    assert captured["url"].endswith("/v2/rerank")
    assert captured["json"]["top_n"] == 2
    assert captured["auth"] == "Bearer test-key"
    assert [item["id"] for item in results] == ["b", "a"]
    assert results[0]["rerank_score"] == 0.98
    assert results[0]["rerank_provider"] == "cohere"


def test_rerank_disabled_returns_original_order() -> None:
    candidates = [
        {"id": "a", "text": "first", "metadata": {}},
        {"id": "b", "text": "second", "metadata": {}},
    ]

    results = rerank_candidates("question", candidates, top_k=2, provider="none")

    assert [item["id"] for item in results] == ["a", "b"]
    assert all(item["rerank_provider"] == "none" for item in results)


def test_rerank_api_error_falls_back_to_original_candidates() -> None:
    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: int) -> FakeResponse:
        return FakeResponse(429, text="rate limited")

    candidates = [
        {"id": "a", "text": "first", "metadata": {}},
        {"id": "b", "text": "second", "metadata": {}},
    ]

    results = rerank_candidates(
        "question",
        candidates,
        top_k=2,
        provider="cohere",
        api_key="test-key",
        request_fn=fake_post,  # type: ignore[arg-type]
    )

    assert [item["id"] for item in results] == ["a", "b"]
    assert "rerank_error" in results[0]


def test_mmr_selects_diverse_non_duplicate_context() -> None:
    candidates = [
        {
            "id": "python_a",
            "text": "Python is a programming language for AI and data science.",
            "metadata": {"heading_path": "Python"},
            "rerank_score": 0.95,
        },
        {
            "id": "python_duplicate",
            "text": "Python is a programming language for AI and data science.",
            "metadata": {"heading_path": "Python"},
            "rerank_score": 0.9,
        },
        {
            "id": "ml",
            "text": "Machine learning uses models, training data and evaluation metrics.",
            "metadata": {"heading_path": "Machine learning"},
            "rerank_score": 0.75,
        },
    ]

    results = select_context_mmr(
        "Python and machine learning",
        candidates,
        final_k=2,
        lambda_mult=0.6,
        min_text_similarity=0.8,
    )

    assert [item["id"] for item in results] == ["python_a", "ml"]
    assert all(item["mmr_selected"] for item in results)
    assert results[0]["mmr_rank"] == 1
    assert results[1]["mmr_rank"] == 2
