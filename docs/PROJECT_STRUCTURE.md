# Project Structure - AIO-S1-014

## Folder Tree

```text
.
├── baseline/
│   ├── README.md
│   └── chatbot_app_native.py
├── docs/
│   ├── backlog.csv
│   ├── baseline_audit.md
│   ├── baseline_vs_upgrade_checklist.md
│   ├── CODEX_WORKFLOW.md
│   ├── GIT_WORKFLOW.md
│   ├── PROJECT_CONTEXT.md
│   ├── PROJECT_SCOPE.md
│   ├── PROJECT_STRUCTURE.md
│   ├── SMOKE_TESTS.md
│   └── weakness_improve.md
├── tests/
│   ├── README.md
│   └── test_sprint1_smoke.py
├── upgrade/
│   ├── .env.example
│   ├── README.md
│   ├── app.py
│   ├── requirements.txt
│   ├── src/
│   │   ├── chunker.py
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── prompts.py
│   │   ├── rag_chain.py
│   │   ├── retriever.py
│   │   ├── vector_store.py
│   │   ├── loaders/
│   │   │   ├── pdf_loader.py
│   │   │   └── notion_loader.py
│   │   └── sync/
│   │       └── notion_sync.py
│   └── tests/
└── README.md
```

## `baseline/`

Purpose:

- Store the original PDF RAG Chatbot implementation.
- Provide a runnable reference version for demo and comparison.
- Remain stable while upgrade work happens elsewhere.

Rule:

- Do not modify `baseline/` unless the active Task ID explicitly requires it.

Run command:

```bash
streamlit run baseline/chatbot_app_native.py
```

## `upgrade/`

Purpose:

- Store the improved Tank Tank Bot implementation.
- Provide modular code for PDF loading, chunking, embedding, vector store, retrieval, prompts, generation, Notion loading and sync.
- Serve as the main area for future technical improvements.

Run placeholder:

```bash
python upgrade/app.py
```

Import smoke check:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## `docs/`

Purpose:

- Store planning, architecture, workflow, backlog and evaluation documents.
- Keep Codex and the developer aligned on scope and acceptance criteria.

## Mapping Baseline To Upgrade Modules

| Baseline feature | Baseline location | Upgrade target |
|---|---|---|
| Streamlit UI | `baseline/chatbot_app_native.py` | `upgrade/app.py` |
| Prompt template | `PROMPT` constant | `upgrade/src/prompts.py` |
| PDF text extraction | `process_pdf()` | `upgrade/src/loaders/pdf_loader.py` |
| Chunking | `chunk_text()` | `upgrade/src/chunker.py` |
| Embedding | `embed()` | `upgrade/src/embeddings.py` |
| ChromaDB storage | `process_pdf()` | `upgrade/src/vector_store.py` |
| Retrieval | `rag()` | `upgrade/src/retriever.py` |
| Generation | `rag()` | `upgrade/src/rag_chain.py` |
| Notion data | Not available | `upgrade/src/loaders/notion_loader.py` and `upgrade/src/sync/notion_sync.py` |

## Rule Summary

- `baseline/`: reference only.
- `upgrade/`: technical implementation work.
- `docs/`: planning and reporting.
- `tests/`: smoke tests and cross-project tests.
- `others/`: ignored local artifacts only; not pushed to GitHub.
