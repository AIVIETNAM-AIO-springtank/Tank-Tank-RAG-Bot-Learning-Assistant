"""Smoke tests for upgrade_new imports."""

from __future__ import annotations

import importlib


def test_core_modules_import() -> None:
    modules = [
        "upgrade_new.app",
        "upgrade_new.src.config",
        "upgrade_new.src.prompts",
        "upgrade_new.src.chunker",
        "upgrade_new.src.embeddings",
        "upgrade_new.src.vector_store",
        "upgrade_new.src.retriever",
        "upgrade_new.src.rag_chain",
        "upgrade_new.src.keyword_store",
        "upgrade_new.src.reranker",
        "upgrade_new.src.memory",
        "upgrade_new.src.evaluation",
        "upgrade_new.src.loaders.pdf_loader",
        "upgrade_new.src.loaders.notion_loader",
        "upgrade_new.src.loaders.notion_block_parser",
        "upgrade_new.src.loaders.image_loader",
        "upgrade_new.src.loaders.ocr_loader",
        "upgrade_new.src.sync.notion_sync",
        "upgrade_new.src.sync.sync_state",
    ]

    for module_name in modules:
        importlib.import_module(module_name)

