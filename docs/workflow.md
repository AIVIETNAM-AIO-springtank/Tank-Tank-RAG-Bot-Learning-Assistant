# Workflow - Tank Tank Bot

## 1. Local Setup

Install dependencies:

```bash
pip install -r upgrade_new/requirements.txt
```

Create local environment file:

```bash
copy upgrade_new/.env.example upgrade_new/.env
```

Required values:

```bash
COHERE_API_KEY=
GEMINI_API_KEY=
NOTION_TOKEN=
NOTION_DATABASE_ID=
```

Recommended feature flags for final demo:

```bash
ENABLE_OCR=true
ENABLE_NOTION_HASH_WRITE=true
NOTION_CONTENT_HASH_PROPERTY=Content Hash
```

## 2. Start Streamlit

From the project root:

```bash
streamlit run upgrade_new/app.py --server.port 8503
```

Open:

```text
http://localhost:8503/
```

Expected sidebar status:

- `OCR/Vision: enabled` if `ENABLE_OCR=true`.
- `Notion hash write: enabled` if `ENABLE_NOTION_HASH_WRITE=true`.
- Retrieval mode default should be `Hybrid RRF`.
- Cohere rerank and MMR should be enabled for the upgraded pipeline.

## 3. PDF Indexing Workflow

1. Upload a PDF in the PDF section.
2. Click `Index PDF`.
3. The app extracts:
   - Text/layout blocks with PyMuPDF.
   - Table units with `pdfplumber` when tables are detected.
   - OCR/Vision units for scanned pages or images when OCR is enabled.
4. The app chunks content and writes embeddings into ChromaDB.
5. Status should show:
   - pages indexed,
   - content units,
   - OCR units,
   - tables,
   - vision errors,
   - chunks,
   - collection count.

Expected behavior:

- Text PDFs should index without OCR.
- Scanned/image-only PDFs require `ENABLE_OCR=true`.
- OCR or table extraction failures should not crash the whole indexing flow.

## 4. Notion Sync Workflow

Use `Sync Notion` for daily incremental sync.

Use `Rebuild Notion Index` when:

- schema/parser logic changed,
- chunking changed,
- empty metadata filtering changed,
- OCR/table extraction changed,
- Chroma contains stale Notion documents.

Full sync flow:

1. Query all Notion database rows.
2. Read database metadata.
3. Fetch each lesson page block tree recursively.
4. Parse blocks into structure-aware units.
5. Filter empty metadata-only documents.
6. Chunk content units.
7. Embed chunks with Cohere.
8. Delete old Notion source from Chroma.
9. Upsert new Notion chunks.
10. Save local sync state.
11. If enabled, write `Content Hash` back to Notion after successful indexing.

Incremental sync flow:

1. Query database rows.
2. Skip unchanged rows using safe local state checks.
3. Fetch and re-index changed/new pages.
4. Delete pages removed from Notion.
5. Save updated local state.

Important guardrail:

- Remote `Content Hash` is only used for safe skip if the Notion `last_edited_time` is not later than the recorded `hash_written_at`.
- If a user edits a page after hash writeback, the system fetches blocks and re-indexes instead of blindly skipping.

## 5. Chat Workflow

1. Sync Notion and/or index PDF.
2. Ask a question in the chat box.
3. Retrieval pipeline:
   - Vector retrieval from Chroma.
   - Keyword retrieval from local BM25.
   - RRF fusion.
   - Optional Cohere rerank.
   - Optional MMR context selection.
4. Gemini generates an answer using only retrieved context.
5. UI displays:
   - answer,
   - citations,
   - Notion source links,
   - debug metadata if enabled.

Expected behavior:

- If the answer is not in indexed documents, the assistant should say it cannot find the information.
- Source previews should not repeat headings excessively.
- Notion citations should not show `empty`, `Emty`, or blank metadata-only rows.

## 6. Evaluation Workflow

Run automated tests:

```bash
python -m pytest tests upgrade_new/tests
```

Run offline evaluation without RAGAS:

```bash
python upgrade_new/scripts/run_ragas_eval.py --testset upgrade_new/eval/testset.example.jsonl --compare-pipelines --formats json,csv,md
```

Run optional RAGAS scoring:

```bash
pip install -r upgrade_new/requirements-eval.txt
python upgrade_new/scripts/run_ragas_eval.py --testset upgrade_new/eval/testset.example.jsonl --compare-pipelines --use-ragas --formats json,csv,md
```

Manual QA checklist:

```text
upgrade_new/docs/manual_testing_questions.md
```

## 7. Final Submission Checklist

- Full tests pass.
- Streamlit starts on `localhost:8503`.
- Notion sync works.
- `Content Hash` is written when hash writeback is enabled.
- PDF text indexing works.
- PDF table extraction has been checked with a table PDF.
- PDF/Notion OCR has been checked with at least one image/scanned sample.
- RAG answers include citations.
- Out-of-document questions do not hallucinate.
- `README.md`, `project_overview.md`, `workflow.md`, and QA report are up to date.
