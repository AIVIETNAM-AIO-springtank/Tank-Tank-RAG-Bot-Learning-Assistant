"""Local BM25 keyword retrieval for hybrid search."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹ_+\-#.]+", re.UNICODE)


class KeywordStore:
    """In-memory BM25 index built from canonical RAG documents."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self.document_tokens: list[list[str]] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.average_document_length = 0.0

    def build(self, documents: list[dict[str, Any]]) -> None:
        """Build a local BM25 index from documents."""
        self.documents = list(documents)
        self.document_tokens = []
        self.term_frequencies = []
        self.document_frequencies = Counter()

        for document in self.documents:
            tokens = tokenize(str(document.get("text") or ""))
            frequencies = Counter(tokens)
            self.document_tokens.append(tokens)
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())

        total_length = sum(len(tokens) for tokens in self.document_tokens)
        self.average_document_length = total_length / len(self.document_tokens) if self.document_tokens else 0.0

    def query(self, question: str, top_k: int) -> list[dict[str, Any]]:
        """Return top keyword matches with BM25 scores."""
        if top_k <= 0 or not question.strip() or not self.documents:
            return []

        query_terms = list(dict.fromkeys(tokenize(question)))
        if not query_terms:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        total_documents = len(self.documents)
        for index, document in enumerate(self.documents):
            score = self._score_document(query_terms, index, total_documents)
            if score <= 0:
                continue
            item = dict(document)
            item["keyword_score"] = score
            item["retrieval_mode"] = "keyword"
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def _score_document(self, query_terms: list[str], index: int, total_documents: int) -> float:
        frequencies = self.term_frequencies[index]
        document_length = len(self.document_tokens[index])
        if not frequencies or document_length == 0:
            return 0.0

        score = 0.0
        average_length = self.average_document_length or 1.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency <= 0:
                continue
            document_frequency = self.document_frequencies.get(term, 0)
            inverse_document_frequency = math.log(
                1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.k1 * (1 - self.b + self.b * document_length / average_length)
            score += inverse_document_frequency * (term_frequency * (self.k1 + 1)) / denominator
        return score


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese/code text and include no-accent variants."""
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    tokens: list[str] = []
    for raw_token in raw_tokens:
        token = raw_token.strip(".,;:!?()[]{}\"'")
        if not token:
            continue
        tokens.append(token)
        accentless = strip_accents(token)
        if accentless and accentless != token:
            tokens.append(accentless)
    return tokens


def strip_accents(text: str) -> str:
    """Return a lowercase accent-insensitive variant of text."""
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")
