"""Streamlit UI for PDF and Notion RAG."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from upgrade_new.src import config
from upgrade_new.src.chunker import chunk_content_units
from upgrade_new.src.embeddings import embed_documents
from upgrade_new.src.loaders.pdf_loader import load_pdf_units
from upgrade_new.src.rag_chain import answer_question
from upgrade_new.src.sync.notion_sync import sync_notion
from upgrade_new.src.utils.errors import UpgradeError, clean_error_message
from upgrade_new.src.vector_store import VectorStore


def init_session_state() -> None:
    """Initialize Streamlit session keys."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("indexed_status", None)
    st.session_state.setdefault("notion_sync_status", None)


def index_uploaded_pdf(uploaded_file: Any, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    """Index an uploaded PDF into persistent ChromaDB."""
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        units = load_pdf_units(temp_path, source_file=uploaded_file.name)
        chunks = chunk_content_units(units, size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            raise UpgradeError("Could not create any chunks from the PDF.")

        document_id = units[0]["metadata"]["document_id"]
        page_numbers = sorted({unit["metadata"]["page_number"] for unit in units})
        embeddings = embed_documents([chunk["text"] for chunk in chunks])
        store = VectorStore()
        store.delete_by_document_id(document_id)
        store.add_documents(chunks, embeddings)

        return {
            "source_file": uploaded_file.name,
            "document_id": document_id,
            "page_count": units[0]["metadata"]["page_count"],
            "indexed_pages": len(page_numbers),
            "content_units": len(units),
            "ocr_units": sum(1 for unit in units if unit.get("metadata", {}).get("ocr_provider")),
            "vision_errors": sum(1 for unit in units if unit.get("metadata", {}).get("ocr_error")),
            "tables_extracted": sum(1 for unit in units if unit.get("metadata", {}).get("block_type") == "table"),
            "chunk_count": len(chunks),
            "collection_count": store.count(),
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def render_sources(sources: list[dict[str, Any]], debug: bool = False) -> None:
    """Render source citations for one assistant answer."""
    if not sources:
        return

    st.caption("Nguon tham khao")
    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata", {})
        title = _source_title(index, metadata)
        distance = source.get("distance")
        if distance is not None:
            title += f" - distance {distance:.4f}"

        with st.expander(title):
            st.write(source.get("text", "")[:1200])
            notion_url = metadata.get("notion_url")
            if notion_url:
                st.link_button("Mo Notion source", notion_url)
            if debug:
                score_fields = {
                    key: source.get(key)
                    for key in ("retrieval_mode", "vector_score", "keyword_score", "hybrid_score", "distance")
                    if source.get(key) is not None
                }
                score_fields.update(
                    {
                        key: source.get(key)
                        for key in (
                            "rrf_score",
                            "vector_rank",
                            "keyword_rank",
                            "rerank_provider",
                            "rerank_model",
                            "rerank_rank",
                            "rerank_score",
                            "rerank_error",
                            "mmr_rank",
                            "mmr_score",
                            "mmr_diversity_penalty",
                            "mmr_duplicate_allowed",
                        )
                        if source.get(key) is not None
                    }
                )
                if score_fields:
                    st.json(score_fields)
                st.json(metadata)


def _source_title(index: int, metadata: dict[str, Any]) -> str:
    source_type = metadata.get("source_type")
    if source_type == "pdf":
        table = metadata.get("table_index")
        table_label = f" - table {table}" if table not in {"", None} else ""
        return (
            f"{index}. PDF - {metadata.get('source_file', 'Unknown PDF')} "
            f"- trang {metadata.get('page_number', '')}{table_label} - chunk {metadata.get('chunk_id', '')}"
        )
    if source_type == "notion":
        granularity = metadata.get("source_granularity", "notion")
        title = metadata.get("title") or "Untitled lesson"
        heading = metadata.get("heading_path") or ""
        week = metadata.get("week") or ""
        module = metadata.get("module") or ""
        detail = heading or f"week {week} module {module}".strip()
        return f"{index}. Notion {granularity} - {title} - {detail}"
    return f"{index}. Source - {source_type or 'unknown'}"


def render_chat_history(debug: bool) -> None:
    """Render previous chat messages with stored citations."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []), debug=debug)


def sidebar_controls() -> dict[str, Any]:
    """Render sidebar controls and return selected settings."""
    with st.sidebar:
        st.subheader("Cau hinh RAG")
        st.caption(f"Generation: `{config.GEMINI_GENERATION_MODEL}`")
        st.caption(f"Embedding: `{config.COHERE_EMBEDDING_MODEL}`")
        if not config.COHERE_API_KEY:
            st.warning("Thieu COHERE_API_KEY.")
        if not config.GEMINI_API_KEY:
            st.warning("Thieu GEMINI_API_KEY.")
        if not config.NOTION_TOKEN or not config.NOTION_DATABASE_ID:
            st.warning("Thieu NOTION_TOKEN hoac NOTION_DATABASE_ID.")
        st.caption(f"OCR/Vision: `{'enabled' if config.ENABLE_OCR else 'disabled'}`")
        st.caption(f"Notion hash write: `{'enabled' if config.ENABLE_NOTION_HASH_WRITE else 'disabled'}`")

        top_k = st.slider("Final context chunks", min_value=1, max_value=10, value=config.DEFAULT_TOP_K)
        retrieval_options = ["hybrid_rrf", "vector", "keyword"]
        configured_mode = "hybrid_rrf" if config.DEFAULT_RETRIEVAL_MODE == "hybrid" else config.DEFAULT_RETRIEVAL_MODE
        default_mode = configured_mode if configured_mode in retrieval_options else "hybrid_rrf"
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            options=retrieval_options,
            index=retrieval_options.index(default_mode),
            format_func=lambda value: {
                "hybrid_rrf": "Hybrid RRF",
                "vector": "Vector",
                "keyword": "Keyword",
            }[value],
        )
        candidate_k = st.slider(
            "Candidate chunks",
            min_value=max(5, int(top_k)),
            max_value=50,
            value=max(config.RERANK_CANDIDATE_K, int(top_k)),
        )
        rerank_enabled = st.toggle("Cohere rerank", value=config.ENABLE_RERANKING)
        rerank_top_n = st.slider(
            "Rerank top-n",
            min_value=max(1, int(top_k)),
            max_value=max(10, int(candidate_k)),
            value=min(max(config.RERANK_TOP_N, int(top_k)), int(candidate_k)),
            disabled=not rerank_enabled,
        )
        mmr_enabled = st.toggle("MMR diversify", value=config.ENABLE_MMR)
        mmr_lambda = st.slider(
            "MMR relevance weight",
            min_value=0.0,
            max_value=1.0,
            value=float(config.MMR_LAMBDA),
            step=0.05,
            disabled=not mmr_enabled,
        )
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=config.CHUNK_SIZE, step=100)
        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=1000,
            value=min(config.CHUNK_OVERLAP, config.CHUNK_SIZE - 1),
            step=50,
        )
        debug = st.toggle("Debug retrieval", value=False)

        st.subheader("PDF")
        uploaded_file = st.file_uploader("Chon PDF", type=["pdf"])
        index_clicked = st.button("Index PDF", type="primary", use_container_width=True)

        st.subheader("Notion")
        notion_sync_clicked = st.button("Sync Notion", use_container_width=True)
        notion_rebuild_clicked = st.button("Rebuild Notion Index", use_container_width=True)

        if st.button("Xoa lich su chat", use_container_width=True):
            st.session_state.chat_history = []

    return {
        "top_k": int(top_k),
        "retrieval_mode": retrieval_mode,
        "candidate_k": int(candidate_k),
        "rerank_enabled": bool(rerank_enabled),
        "rerank_top_n": int(rerank_top_n),
        "mmr_enabled": bool(mmr_enabled),
        "mmr_lambda": float(mmr_lambda),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "debug": debug,
        "uploaded_file": uploaded_file,
        "index_clicked": index_clicked,
        "notion_sync_clicked": notion_sync_clicked,
        "notion_rebuild_clicked": notion_rebuild_clicked,
    }


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title=config.APP_TITLE, layout="wide")
    init_session_state()
    settings = sidebar_controls()

    st.title(config.APP_TITLE)
    st.caption("PDF + Notion RAG with structure-aware chunking, ChromaDB, Cohere and Gemini.")

    if settings["chunk_overlap"] >= settings["chunk_size"]:
        st.error("Chunk overlap must be smaller than chunk size.")
        return

    if settings["index_clicked"]:
        if settings["uploaded_file"] is None:
            st.warning("Hay chon mot file PDF truoc.")
        else:
            with st.spinner("Reading PDF blocks, chunking, embedding and saving to ChromaDB..."):
                try:
                    status = index_uploaded_pdf(
                        settings["uploaded_file"],
                        chunk_size=settings["chunk_size"],
                        chunk_overlap=settings["chunk_overlap"],
                    )
                    st.session_state.indexed_status = status
                    st.session_state.chat_history = []
                    st.success(f"Indexed {status['chunk_count']} chunks from {status['indexed_pages']} PDF pages.")
                except Exception as exc:
                    st.error(clean_error_message(exc))

    if settings["notion_sync_clicked"]:
        with st.spinner("Incrementally syncing changed Notion pages..."):
            try:
                summary = sync_notion(
                    chunk_size=settings["chunk_size"],
                    chunk_overlap=settings["chunk_overlap"],
                    sync_mode="incremental",
                )
                st.session_state.notion_sync_status = summary
                st.session_state.chat_history = []
                st.success(
                    f"Synced Notion incrementally: new {summary['pages_new']}, "
                    f"changed {summary['pages_changed']}, skipped {summary['pages_skipped']}, "
                    f"indexed {summary['chunks_indexed']} documents/chunks."
                )
            except Exception as exc:
                st.error(clean_error_message(exc))

    if settings["notion_rebuild_clicked"]:
        with st.spinner("Rebuilding all Notion pages, embeddings and local sync state..."):
            try:
                summary = sync_notion(
                    chunk_size=settings["chunk_size"],
                    chunk_overlap=settings["chunk_overlap"],
                    sync_mode="full",
                )
                st.session_state.notion_sync_status = summary
                st.session_state.chat_history = []
                st.success(
                    f"Rebuilt Notion index: pages {summary['pages_seen']}, "
                    f"indexed {summary['chunks_indexed']} documents/chunks."
                )
            except Exception as exc:
                st.error(clean_error_message(exc))

    pdf_status = st.session_state.indexed_status
    notion_status = st.session_state.notion_sync_status

    if pdf_status:
        st.info(
            f"Current PDF: {pdf_status['source_file']} | "
            f"Pages indexed: {pdf_status['indexed_pages']}/{pdf_status['page_count']} | "
            f"Blocks: {pdf_status['content_units']} | "
            f"OCR units: {pdf_status.get('ocr_units', 0)} | "
            f"Tables: {pdf_status.get('tables_extracted', 0)} | "
            f"Vision errors: {pdf_status.get('vision_errors', 0)} | "
            f"Chunks: {pdf_status['chunk_count']} | "
            f"Collection count: {pdf_status['collection_count']}"
        )

    if notion_status:
        st.info(
            f"Notion sync ({notion_status.get('sync_mode', 'unknown')}): pages {notion_status['pages_seen']} | "
            f"new {notion_status.get('pages_new', 0)} | "
            f"changed {notion_status.get('pages_changed', 0)} | "
            f"skipped {notion_status.get('pages_skipped', 0)} | "
            f"hash unchanged {notion_status.get('pages_hash_unchanged', 0)} | "
            f"removed {notion_status.get('pages_removed', 0)} | "
            f"metadata docs {notion_status['metadata_documents']} | "
            f"metadata skipped empty {notion_status.get('metadata_documents_skipped_empty', 0)} | "
            f"content units {notion_status['content_documents']} | "
            f"OCR units {notion_status.get('ocr_units', 0)} | "
            f"vision errors {notion_status.get('vision_errors', 0)} | "
            f"tables {notion_status.get('tables_extracted', 0)} | "
            f"hash written {notion_status.get('hash_written', 0)} | "
            f"hash write failed {notion_status.get('hash_write_failed', 0)} | "
            f"indexed {notion_status['chunks_indexed']} | "
            f"skipped empty {notion_status['skipped_empty_pages']}"
        )

    if not pdf_status and not notion_status:
        st.info("Upload/index PDF or sync Notion to start chatting.")

    render_chat_history(debug=settings["debug"])

    question = st.chat_input("Nhap cau hoi ve PDF/Notion da index...", disabled=not bool(pdf_status or notion_status))
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and calling Gemini..."):
            try:
                result = answer_question(
                    question,
                    top_k=settings["top_k"],
                    vector_store=VectorStore(),
                    retrieval_mode=settings["retrieval_mode"],
                    candidate_k=settings["candidate_k"],
                    rerank_enabled=settings["rerank_enabled"],
                    rerank_top_n=settings["rerank_top_n"],
                    mmr_enabled=settings["mmr_enabled"],
                    mmr_lambda=settings["mmr_lambda"],
                )
                answer = result["answer"]
                sources = result["sources"]
                st.write(answer)
                if settings["debug"]:
                    st.json(result.get("retrieval_debug", {}))
                render_sources(sources, debug=settings["debug"])
            except Exception as exc:
                answer = clean_error_message(exc)
                sources = []
                st.error(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})


if __name__ == "__main__":
    main()
