"""Tests for local BM25 keyword retrieval."""

from __future__ import annotations

from upgrade_new.src.keyword_store import KeywordStore, tokenize


def test_tokenize_keeps_vietnamese_and_accentless_variants() -> None:
    tokens = tokenize("Giới thiệu Python match-case range()")

    assert "giới" in tokens
    assert "gioi" in tokens
    assert "python" in tokens
    assert "match-case" in tokens
    assert "range" in tokens


def test_keyword_store_ranks_exact_keyword_match_first() -> None:
    documents = [
        {"id": "doc1", "text": "General Python branching content", "metadata": {"source_type": "notion"}},
        {"id": "doc2", "text": "Python 3.10 match-case handles branching", "metadata": {"source_type": "notion"}},
    ]
    store = KeywordStore()
    store.build(documents)

    results = store.query("match-case", top_k=2)

    assert results[0]["id"] == "doc2"
    assert results[0]["keyword_score"] > 0
    assert results[0]["retrieval_mode"] == "keyword"
