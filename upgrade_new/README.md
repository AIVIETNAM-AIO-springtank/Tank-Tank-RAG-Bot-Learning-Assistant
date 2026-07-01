# Upgrade New - Tank Tank Bot

`upgrade_new/` is the final upgrade workspace for Tank Tank Bot, a RAG assistant for AIO 2026 learning materials.
It is separate from `baseline/`, which remains reference-only.

## What It Does

- Streamlit chat UI for PDF + Notion learning materials.
- PDF ingestion with PyMuPDF layout blocks, optional Gemini Vision OCR, and optional `pdfplumber` table extraction.
- Notion database sync with flexible metadata parsing, recursive block parsing, table/image support, local incremental sync state, and optional `Content Hash` writeback.
- Structure-aware chunking: page-aware, block-aware, heading-aware, paragraph/sentence recursive split, metadata enrichment.
- ChromaDB persistent vector store with Cohere multilingual embeddings.
- Retrieval pipeline: vector + BM25 keyword search, Reciprocal Rank Fusion, optional Cohere Rerank, and MMR context selection.
- Gemini generation with source citations.
- Offline evaluation reports with JSON/CSV/Markdown outputs and optional RAGAS metrics.

## Setup

```bash
pip install -r upgrade_new/requirements.txt
copy upgrade_new/.env.example upgrade_new/.env
```

Set these values in `.env`, environment variables, or Streamlit secrets:

```bash
COHERE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
GEMINI_API_KEYS=optional_key_1,optional_key_2
NOTION_TOKEN=your_key_here
NOTION_DATABASE_ID=your_database_id_here
```

Optional feature flags:

```bash
ENABLE_OCR=true
ENABLE_NOTION_HASH_WRITE=true
NOTION_CONTENT_HASH_PROPERTY=Content Hash
```

For Notion hash writeback, the Notion integration must have `Update content`
permission and the database must contain a rich text property named `Content Hash`.

## Run

From the project root:

```bash
streamlit run upgrade_new/app.py --server.port 8503
```

Then open:

```text
http://localhost:8503/
```

## Test

```bash
python -m pytest tests upgrade_new/tests
```

Manual QA checklist:

```text
upgrade_new/docs/manual_testing_questions.md
```

## Notion Sync

- `Sync Notion`: incremental sync for changed/new/removed pages.
- `Rebuild Notion Index`: full Notion re-index and local sync state refresh.
- Empty metadata-only Notion rows such as `empty`, `Emty`, blank title, or `Untitled Notion lesson` are not indexed.
- If `ENABLE_NOTION_HASH_WRITE=true`, content hash is written back after successful indexing.
- If hash writeback fails, indexing still completes and the UI shows `hash write failed`.

## Evaluation

Generate report files without RAGAS scoring:

```bash
python upgrade_new/scripts/run_ragas_eval.py --testset upgrade_new/eval/testset.example.jsonl --formats json,csv,md
```

Compare retrieval pipelines:

```bash
python upgrade_new/scripts/run_ragas_eval.py --testset upgrade_new/eval/testset.example.jsonl --compare-pipelines --formats json,csv,md
```

Optional RAGAS metrics:

```bash
pip install -r upgrade_new/requirements-eval.txt
python upgrade_new/scripts/run_ragas_eval.py --testset upgrade_new/eval/testset.example.jsonl --compare-pipelines --use-ragas
```

Reports are written to `upgrade_new/eval/reports/`.

## Current Limits

- Gemini Vision OCR depends on image accessibility and API quota.
- PDF table extraction depends on `pdfplumber`; complex tables may still need manual validation.
- Chat memory, HyDE, and query decomposition are intentionally left for later phases.
- The app is designed for local/Streamlit deployment, not multi-user production hardening.
