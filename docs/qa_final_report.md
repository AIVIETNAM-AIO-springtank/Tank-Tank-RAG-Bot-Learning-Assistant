# QA Final Report - Tank Tank Bot

Generated: 2026-06-28

## Executive Summary

The `upgrade_new/` project is feature-complete for the current upgrade MVP: PDF + Notion RAG, structure-aware chunking, hybrid retrieval, rerank/MMR, evaluation reports, OCR/Vision, advanced table extraction, and optional Notion hash writeback.

This QA pass reviewed code logic, tests, docs, runtime dependencies, Chroma state, and final submission readiness. One important sync bug was found and fixed during this pass.

Current status: ready for manual demo/submission after the manual QA checklist is run on the target data samples.

## Scope Reviewed

- Streamlit UI and status reporting.
- PDF ingestion, OCR/Vision, table extraction.
- Notion metadata/page parsing, table/image handling.
- Notion full/incremental sync, local state, hash writeback.
- Chunking and metadata preservation.
- Vector store, keyword store, retrieval, rerank, MMR.
- RAG generation guardrails and citations.
- Offline evaluation scripts and existing reports.
- Project docs and final workflow docs.

## Automated Evidence

Commands run:

```bash
python -B -m pytest upgrade_new\tests\test_notion_sync.py --basetemp C:\tmp\pytest-qa-notion-sync -o cache_dir=C:\tmp\pytest-cache-qa-notion-sync
```

Result:

```text
11 passed
```

Full regression:

```bash
python -B -m pytest tests upgrade_new\tests --basetemp C:\tmp\pytest-qa-final -o cache_dir=C:\tmp\pytest-cache-qa-final
```

Result:

```text
80 passed
```

Runtime dependency check:

```text
pdfplumber_installed=True
```

Streamlit status:

```text
localhost:8503 listen pid=26856
```

Chroma state check:

```text
collection_count=294
notion_metadata_documents=31
placeholder_metadata_documents=0
```

## Issue Found And Fixed

### Safe Notion Remote Hash Skip

Finding:

The incremental sync logic could skip a Notion page whenever remote `Content Hash` matched local `content_hash`. That is not always safe, because a user can edit page content after the hash was written. In that case, the remote hash property may still hold the old hash while `last_edited_time` has advanced.

Fix:

Remote hash skip now requires:

- remote `Content Hash` equals local `content_hash`;
- local state has `hash_written_at`;
- Notion `last_edited_time <= hash_written_at`.

If `last_edited_time` is later than `hash_written_at`, the system fetches blocks and re-indexes.

Regression tests added:

- skip is allowed when the change was caused by prior hash writeback;
- skip is not allowed when the page was edited after hash writeback.

### Full Sync Hash Writeback Scope

Finding:

Full sync wrote hash back for every loaded lesson, including pages with no indexed documents.

Fix:

Full sync now writes hash only for lessons that produced at least one metadata/content document for indexing.

## Code Logic QA Notes

### PDF Ingestion

Status: good for MVP.

- PyMuPDF extracts layout-aware text blocks.
- Scanned/no-text pages can use Gemini Vision when `ENABLE_OCR=true`.
- Embedded image blocks can be OCRed within `VISION_MAX_IMAGES_PER_DOC`.
- `pdfplumber` is installed and used for table units.
- If OCR/table extraction fails, ingestion degrades without crashing the full flow.

Remaining risk:

- Complex tables may still need manual inspection.
- OCR quality depends on image quality and Gemini quota.

### Notion Ingestion

Status: good for MVP.

- Flexible schema detection works for title and common metadata fields.
- Empty metadata-only documents are filtered.
- Content blocks are parsed recursively.
- Tables become markdown table units.
- Images keep caption/URL and can include OCR text when enabled.
- `Content Hash` is read and can be written back after indexing.

Remaining risk:

- Notion file URLs may expire or be inaccessible to Gemini Vision.
- Very large Notion pages may require slower sync due OCR calls.

### Chunking

Status: good for MVP.

- Page-aware PDF chunking avoids cross-page citations.
- Heading context is prepended once per chunk, avoiding the repeated-heading bug.
- Code/equation/table/image block types keep useful metadata.
- Metadata fields are preserved into Chroma.

Remaining risk:

- Very long code/table blocks may still require manual tuning of chunk size/overlap.

### Retrieval And Generation

Status: good for MVP.

- Default recommended pipeline is `hybrid_rrf + Cohere rerank + MMR`.
- RRF avoids mixing incompatible vector and BM25 score scales.
- MMR reduces repeated contexts.
- Gemini prompt restricts answers to retrieved context.
- Citations show PDF/Notion source metadata.

Remaining risk:

- Answer relevancy in prior RAGAS report was moderate, so final answer style can still be tuned later.

## Manual QA Checklist

Manual test file:

```text
upgrade_new/docs/manual_testing_questions.md
```

Minimum manual pass before submission:

1. Rebuild Notion index and verify `hash written > 0`.
2. Open Notion and confirm `Content Hash` is populated.
3. Run incremental sync again and confirm unchanged pages are skipped.
4. Ask Basic Python questions and confirm no repeated heading bug.
5. Ask week/module metadata questions and confirm no `empty`/`Emty` citation.
6. Upload a table PDF and confirm table citation.
7. Upload a scanned/image PDF and confirm OCR units or graceful vision errors.
8. Ask an out-of-document question and confirm no hallucinated answer.

Manual QA not fully automated:

- Image OCR accuracy.
- Complex PDF table quality.
- Live Notion hash writeback on all rows.
- End-user answer quality for broad summary questions.

## Evaluation Status

Existing grounded report:

- `upgrade_new/eval/reports/ragas_grounded_report.md`
- `upgrade_new/eval/reports/ragas_grounded_report.csv`
- `upgrade_new/eval/reports/ragas_grounded_report.json`
- `upgrade_new/eval/reports/ragas_grounded_analysis.md`

Prior result summary:

- Vector only: faithfulness `0.958`, answer relevancy `0.725`, context precision `0.972`, context recall `0.750`.
- Hybrid RRF + Cohere Rerank + MMR: faithfulness `1.000`, answer relevancy `0.729`, context precision `1.000`, context recall `1.000`.

Interpretation:

The upgraded retrieval pipeline is stronger than vector-only on the grounded testset. The remaining quality gap is mainly answer style/relevancy and testset size.

## Final Readiness

Ready:

- Core feature scope is implemented.
- Tests pass.
- Streamlit runs.
- `pdfplumber` dependency is installed.
- Chroma has no empty Notion metadata citations.
- Docs now include project overview, workflow, manual QA, and this QA report.

Recommended before public submission:

- Run the manual QA checklist with one real table PDF and one real scanned/image PDF.
- Run one live Notion rebuild with hash writeback and inspect a few rows.
- If screenshots are needed for the submission, capture:
  - Streamlit main UI,
  - Notion sync status,
  - source citation/debug panel,
  - Notion `Content Hash` column populated.

Deferred after submission:

- Larger evaluation testset.
- Per-source recall metrics.
- Query decomposition/HyDE.
- Chat memory/follow-up rewrite.
- Production deployment hardening.
