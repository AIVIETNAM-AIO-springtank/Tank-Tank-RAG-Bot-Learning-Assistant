# Baseline vs Upgrade Checklist - AIO-S1-015

Use this checklist during Sprint 5 demo preparation to compare the original baseline with the upgraded assistant.

| Capability | Baseline | Upgrade target | Demo status |
|---|---|---|---|
| PDF upload | Present | Keep and improve | Pending Sprint 2 implementation |
| PDF text extraction | `pypdf` plain text | Page-level loader with metadata | Pending |
| Chunking | Fixed text chunks | Metadata-aware chunks | Pending |
| Persistent DB | In-memory ChromaDB | `PersistentClient` with stable path | Pending |
| Metadata/source | Missing | `source_file`, `page_number`, `chunk_id`, `document_id` | Pending |
| Citation display | Missing | Show source list and chunk preview | Pending |
| Error handling | Minimal | Friendly errors for PDF/Ollama/model issues | Pending |
| Chat memory | Display-only history | Short message history or follow-up condensation | Pending |
| Model config | Constants in app | Config/env/sidebar driven | Skeleton present |
| Notion CSV | Not available | CSV export loader | Skeleton present |
| Notion API sync | Not available | Incremental sync module | Skeleton present |
| Evaluation | Not available | Golden question set and smoke tests | Sprint 1 smoke present |
| Folder structure | Single native app | `baseline/`, `upgrade/`, `docs/`, `tests/` | Present |

## Demo Notes

- Start with baseline to show the original assignment flow.
- Then show upgrade structure and explain planned improvements.
- For completed features, demonstrate behavior directly.
- For pending features, point to the matching backlog Task ID rather than claiming completion.

## Minimum Sprint 5 Acceptance

- Baseline can still be explained and run as the reference version.
- Upgrade demonstrates the agreed improvements that were completed by deadline.
- Known limitations are documented honestly.
- No secrets, local vector database or ignored `others/` artifacts are included in the submitted repo.

