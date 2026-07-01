# Project Context - Tank Tank Bot

## 1. Purpose

This document is the single source of truth for Codex and project development. Every task should follow this context unless a newer approved planning document or Task ID explicitly changes the scope.

- Task ID: `AIO-S1-002`
- Epic: `EPIC-01` - Architecture, Baseline & Upgrade Separation
- Priority: P0
- Sprint: Sprint 1 - Architecture, Baseline & Upgrade
- Deadline: `2026-06-21`

## 2. Product Vision

Tank Tank Bot is a learning assistant built with Retrieval Augmented Generation (RAG). The assistant helps AIO learners ask questions over learning materials and receive Vietnamese answers grounded in source documents.

The project is not only a simple PDF chatbot. The final direction is a structured learning assistant that can compare an original baseline implementation with an improved upgrade version.

Core product goals:

- Help learners search and understand AIO learning materials faster.
- Keep answers grounded in retrieved context instead of unsupported model knowledge.
- Show citations/sources so users can verify answers.
- Keep the implementation explainable for coursework, demo and review.
- Maintain a clear separation between the original baseline and the improved version.

## 3. Project Versions

### 3.1. `baseline/` - PDF RAG Chatbot

`baseline/` is the original PDF RAG Chatbot version based on the Project 1.2 assignment.

Purpose:

- Preserve the original implementation for comparison.
- Provide a simple demo of the initial PDF RAG flow.
- Act as the reference version when explaining weaknesses and improvements.

Expected baseline behavior:

- Upload a PDF.
- Extract text with `pypdf`.
- Chunk text.
- Create embeddings with Ollama `bge-m3`.
- Store vectors in ChromaDB.
- Retrieve relevant chunks.
- Generate answers using Ollama LLM, originally `vicuna:7b-v1.5-q5_1`.
- Display chat through Streamlit.

Rules:

- Do not modify `baseline/` unless the current Task ID explicitly requires it.
- Do not mix upgrade-only logic into baseline.
- Baseline should remain easy to run and compare against upgrade.

### 3.2. `upgrade/` - Improved Tank Tank Bot

`upgrade/` is the improved implementation built from the baseline flow.

Target improvements:

- PDF ingestion with page/source metadata.
- Notion database or CSV export ingestion.
- Persistent ChromaDB storage.
- Source citation in answers.
- Short conversation memory for follow-up questions.
- Configurable model and RAG parameters.
- Better error handling for PDF, Ollama and model failures.
- Modular code structure that is easier to test and explain.

Rules:

- Technical improvements belong in `upgrade/`.
- Code should be readable, typed where practical and documented with concise docstrings.
- Do not hard-code tokens, API keys or personal credentials.
- Preserve a clear path from baseline flow to upgrade modules.

## 4. Tech Stack

Primary stack:

- Streamlit: web UI for upload, chat, controls and demo.
- `pypdf`: baseline PDF text extraction.
- ChromaDB: vector database.
- Ollama: local embedding and LLM runtime.
- `bge-m3`: default embedding model.
- `vicuna:7b-v1.5-q5_1`: original baseline LLM model.

Upgrade stack expectations:

- LLM model should be configurable where possible.
- Acceptable alternative local models include Qwen, Llama or Gemma variants if available through Ollama.
- Persistent vector storage should use `chromadb.PersistentClient`.
- Configuration should live in config files or environment variables, not scattered constants.

## 5. Data Sources

### 5.1. PDF

PDF is the mandatory MVP data source because it matches the original assignment.

Expected metadata:

- Source file name.
- Page number or page range.
- Chunk ID.
- Document ID if file hashing is implemented.

### 5.2. Notion Database / CSV Export

Notion is an upgrade data source for turning the project into Tank Tank Bot.

Supported target paths:

- MVP/P1: Notion CSV export.
- P2: Notion API database sync and incremental sync.

Expected metadata:

- Title.
- Week.
- Date.
- Module.
- Lecturer.
- Label.
- Notion URL or page ID.
- Last edited time when available.

## 6. Architecture Target

The target architecture uses three main areas:

```text
baseline/
  Original PDF RAG Chatbot implementation.
  Read-only/reference unless a task explicitly targets baseline.

upgrade/
  Improved implementation.
  Contains app entrypoint, modular RAG code, loaders, vector store, retriever, generation logic and tests.

docs/
  Project planning, architecture notes, backlog support docs, demo scripts and evaluation notes.
```

Current workspace note: the repository uses `baseline/` and `upgrade/` as the active project folders. Legacy folders and unrelated artifacts are stored under `others/`.

## 7. Coding Rules

General rules:

1. Work on one Task ID at a time.
2. Before coding, state the Task ID, goal, expected files and acceptance criteria.
3. Do not modify baseline/reference code unless the task explicitly requires it.
4. Put technical improvements in `upgrade/`.
5. Put planning and documentation outputs in `docs/`.
6. Keep changes scoped to the current Task ID.
7. Do not hard-code tokens, API keys, secrets or personal credentials.
8. Use environment variables or `.env.example` placeholders for sensitive configuration.
9. Write clear code with concise docstrings for public functions/modules.
10. Keep code explainable for coursework submission and demo.
11. Prefer simple, reliable implementations over advanced features that risk the deadline.
12. After coding, report files changed, test method and acceptance criteria status.

RAG-specific rules:

1. Answers should be grounded in retrieved context.
2. If context is missing or weak, prefer saying that the information was not found.
3. Preserve metadata from source documents through chunking, indexing, retrieval and UI.
4. Show citations/sources whenever retrieval succeeds.
5. Keep model names and retrieval parameters configurable where practical.

## 8. Definition Of Done

A task is done only when all relevant items below are satisfied:

- The implementation or document matches the selected Task ID.
- Scope does not exceed the backlog item without explicit approval.
- Baseline remains unchanged unless the task explicitly targets it.
- Upgrade changes are placed under `upgrade/` or the current physical equivalent.
- Documentation changes are placed under `docs/`.
- No secrets, tokens or personal credentials are committed.
- New or changed code has clear names and concise docstrings where useful.
- The work is tested or validated with a clear command/checklist.
- Acceptance criteria from the backlog row are explicitly checked.
- The final report lists files changed, tests run and remaining limitations if any.

## 9. Current Scope Guardrails

Before `2026-06-30`, the project should prioritize:

- Baseline and upgrade separation.
- Reliable PDF RAG core.
- Persistent vector database.
- Metadata and source citation.
- Basic chat memory.
- Notion CSV/API support only after core RAG is stable.
- Evaluation, documentation and demo readiness.

The project should not prioritize:

- Production authentication.
- Complex multi-user infrastructure.
- Cloud deployment as a required deliverable.
- Fine-tuning LLMs.
- Perfect OCR/table extraction for every PDF type.
- Large scope additions outside the approved backlog.
