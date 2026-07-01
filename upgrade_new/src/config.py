"""Central configuration for the new upgrade implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

if load_dotenv:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPGRADE_ROOT = Path(__file__).resolve().parents[1]


def _streamlit_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


def get_setting(name: str, default: str = "") -> str:
    """Read a setting from environment first, then Streamlit secrets."""
    return os.getenv(name) or _streamlit_secret(name, default)


def get_int_setting(name: str, default: int) -> int:
    """Read an integer setting with a safe fallback."""
    value = get_setting(name, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_float_setting(name: str, default: float) -> float:
    """Read a float setting with a safe fallback."""
    value = get_setting(name, str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_bool_setting(name: str, default: bool = False) -> bool:
    """Read a boolean setting from common string forms."""
    value = get_setting(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def get_csv_setting(name: str) -> list[str]:
    """Read a comma-separated setting as a clean list."""
    value = get_setting(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


APP_TITLE = get_setting("APP_TITLE", "Tank Tank Bot")
GEMINI_GENERATION_MODEL = get_setting("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = get_setting("GEMINI_VISION_MODEL", "gemini-2.5-flash")
COHERE_EMBEDDING_MODEL = get_setting("COHERE_EMBEDDING_MODEL", "embed-multilingual-v3.0")
EMBEDDING_DIM = get_int_setting("EMBEDDING_DIM", 1024)
CHROMA_PATH = get_setting("CHROMA_PATH", str(UPGRADE_ROOT / "chroma_db"))
COLLECTION_NAME = get_setting("COLLECTION_NAME", "aio2026_learning_assistant")
DEFAULT_TOP_K = get_int_setting("DEFAULT_TOP_K", 4)
DEFAULT_RETRIEVAL_MODE = get_setting("DEFAULT_RETRIEVAL_MODE", "hybrid_rrf")
HYBRID_CANDIDATE_K = get_int_setting("HYBRID_CANDIDATE_K", 30)
HYBRID_VECTOR_WEIGHT = get_float_setting("HYBRID_VECTOR_WEIGHT", 0.7)
HYBRID_KEYWORD_WEIGHT = get_float_setting("HYBRID_KEYWORD_WEIGHT", 0.3)
RRF_K = get_int_setting("RRF_K", 60)
CHUNK_SIZE = get_int_setting("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = get_int_setting("CHUNK_OVERLAP", 200)
REQUEST_TIMEOUT = get_int_setting("REQUEST_TIMEOUT", 60)
BATCH_SIZE = get_int_setting("BATCH_SIZE", 32)
NOTION_TOKEN = get_setting("NOTION_TOKEN", "")
NOTION_DATABASE_ID = get_setting("NOTION_DATABASE_ID", "")
COHERE_API_KEY = get_setting("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = get_setting("COHERE_RERANK_MODEL", "rerank-v3.5")
RERANK_PROVIDER = get_setting("RERANK_PROVIDER", "cohere" if COHERE_API_KEY else "none").strip().lower()
RERANK_CANDIDATE_K = get_int_setting("RERANK_CANDIDATE_K", 30)
RERANK_TOP_N = get_int_setting("RERANK_TOP_N", 10)
FINAL_CONTEXT_K = get_int_setting("FINAL_CONTEXT_K", DEFAULT_TOP_K)
MMR_LAMBDA = get_float_setting("MMR_LAMBDA", 0.65)
MMR_MIN_TEXT_SIMILARITY = get_float_setting("MMR_MIN_TEXT_SIMILARITY", 0.82)
GEMINI_API_KEY = get_setting("GEMINI_API_KEY", "") or get_setting("GOOGLE_API_KEY", "")
GEMINI_API_KEYS = get_csv_setting("GEMINI_API_KEYS") or ([GEMINI_API_KEY] if GEMINI_API_KEY else [])
GENERATION_PROVIDER = get_setting("GENERATION_PROVIDER", "gemini").strip().lower()
COHERE_GENERATION_MODEL = get_setting("COHERE_GENERATION_MODEL", "command-r7b-12-2024")
ENABLE_HYBRID_SEARCH = get_bool_setting("ENABLE_HYBRID_SEARCH", False)
ENABLE_RERANKING = get_bool_setting("ENABLE_RERANKING", RERANK_PROVIDER == "cohere" and bool(COHERE_API_KEY))
ENABLE_MMR = get_bool_setting("ENABLE_MMR", True)
ENABLE_OCR = get_bool_setting("ENABLE_OCR", False)
VISION_MAX_IMAGES_PER_DOC = get_int_setting("VISION_MAX_IMAGES_PER_DOC", 20)
VISION_IMAGE_DPI = get_int_setting("VISION_IMAGE_DPI", 150)
ENABLE_CONTENT_HASH_SYNC = get_bool_setting("ENABLE_CONTENT_HASH_SYNC", False)
ENABLE_NOTION_HASH_WRITE = get_bool_setting("ENABLE_NOTION_HASH_WRITE", False)
NOTION_CONTENT_HASH_PROPERTY = get_setting("NOTION_CONTENT_HASH_PROPERTY", "Content Hash")
NOTION_HASH_WRITE_MODE = get_setting("NOTION_HASH_WRITE_MODE", "after_successful_index")


def as_dict() -> dict[str, Any]:
    """Return non-secret config values for debug display."""
    return {
        "app_title": APP_TITLE,
        "gemini_generation_model": GEMINI_GENERATION_MODEL,
        "gemini_vision_model": GEMINI_VISION_MODEL,
        "gemini_api_key_count": len(GEMINI_API_KEYS),
        "generation_provider": GENERATION_PROVIDER,
        "cohere_generation_model": COHERE_GENERATION_MODEL,
        "cohere_embedding_model": COHERE_EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chroma_path": CHROMA_PATH,
        "collection_name": COLLECTION_NAME,
        "default_top_k": DEFAULT_TOP_K,
        "default_retrieval_mode": DEFAULT_RETRIEVAL_MODE,
        "hybrid_candidate_k": HYBRID_CANDIDATE_K,
        "hybrid_vector_weight": HYBRID_VECTOR_WEIGHT,
        "hybrid_keyword_weight": HYBRID_KEYWORD_WEIGHT,
        "rrf_k": RRF_K,
        "cohere_rerank_model": COHERE_RERANK_MODEL,
        "rerank_provider": RERANK_PROVIDER,
        "rerank_candidate_k": RERANK_CANDIDATE_K,
        "rerank_top_n": RERANK_TOP_N,
        "final_context_k": FINAL_CONTEXT_K,
        "mmr_lambda": MMR_LAMBDA,
        "mmr_min_text_similarity": MMR_MIN_TEXT_SIMILARITY,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "request_timeout": REQUEST_TIMEOUT,
        "batch_size": BATCH_SIZE,
        "enable_hybrid_search": ENABLE_HYBRID_SEARCH,
        "enable_reranking": ENABLE_RERANKING,
        "enable_mmr": ENABLE_MMR,
        "enable_ocr": ENABLE_OCR,
        "vision_max_images_per_doc": VISION_MAX_IMAGES_PER_DOC,
        "vision_image_dpi": VISION_IMAGE_DPI,
        "enable_content_hash_sync": ENABLE_CONTENT_HASH_SYNC,
        "enable_notion_hash_write": ENABLE_NOTION_HASH_WRITE,
        "notion_content_hash_property": NOTION_CONTENT_HASH_PROPERTY,
        "notion_hash_write_mode": NOTION_HASH_WRITE_MODE,
    }
