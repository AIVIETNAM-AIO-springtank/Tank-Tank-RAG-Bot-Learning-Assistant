# Baseline Audit - AIO-S1-003

## Scope

This audit covers the current baseline implementation in `baseline/chatbot_app_native.py`.
No baseline code was refactored for this task.

## Baseline Purpose

The baseline is the original PDF RAG Chatbot version used for comparison with the upgraded Tank Tank Bot.
It demonstrates the minimum assignment flow: upload PDF, extract text, chunk, embed, store vectors, retrieve context and generate an answer.

## Main Functions And Components

| Component | Role |
|---|---|
| `LLM_MODEL` | Defines the local Ollama generation model, currently `vicuna:7b-v1.5-q5_1`. |
| `EMBED_MODEL` | Defines the local Ollama embedding model, currently `bge-m3`. |
| `PROMPT` | Prompt template that asks the LLM to answer in Vietnamese and avoid fabrication when context is missing. |
| `embed(texts)` | Calls `ollama.embed()` and returns embeddings. |
| `chunk_text(text, size=1000, overlap=200)` | Splits extracted text into overlapping chunks. |
| `process_pdf(uploaded_file)` | Saves upload temporarily, extracts PDF text, chunks it, embeds chunks and stores them in ChromaDB. |
| `rag(question, collection, k=4)` | Embeds the question, retrieves top-k chunks, builds context and calls `ollama.chat()`. |
| Streamlit UI | Provides PDF upload, processing button, chat input and chat history display. |

## Runtime Flow

1. User opens the Streamlit app.
2. User uploads a PDF from the sidebar.
3. `process_pdf()` writes the uploaded file to a temporary PDF.
4. `pypdf.PdfReader` extracts text from each page.
5. `chunk_text()` splits the full text into chunks.
6. `embed()` sends chunks to Ollama `bge-m3`.
7. ChromaDB stores chunk documents and vectors in an in-memory collection.
8. User asks a question through `st.chat_input()`.
9. `rag()` embeds the question and retrieves `k=4` similar chunks.
10. Retrieved chunks are joined into context and sent to the LLM.
11. The answer is displayed in Streamlit and appended to `st.session_state.chat_history`.

## How To Run Baseline Demo

Prerequisites:

- Python environment with dependencies installed.
- Ollama running locally.
- Required models pulled:
  - `ollama pull vicuna:7b-v1.5-q5_1`
  - `ollama pull bge-m3`

Command:

```bash
streamlit run baseline/chatbot_app_native.py
```

Expected demo behavior:

- Upload a text-based PDF.
- Click the processing button.
- Ask a question related to the PDF.
- Receive a Vietnamese answer based on retrieved context.

## Baseline Weaknesses

| Weakness | Impact |
|---|---|
| Uses `pypdf` plain text extraction only | Tables, images, scanned pages and complex layouts may be lost or malformed. |
| Chunks are detached from page metadata | The app cannot show page-level citations. |
| Uses `chromadb.Client()` in memory | Vector database is lost after app restart. |
| Collection name uses timestamp | Uploading the same file repeatedly creates duplicate collections. |
| Retrieval uses vector search only with fixed `k=4` | Keyword-heavy or broad questions may retrieve weak context. |
| Chat history is display-only | Follow-up questions are not grounded in prior conversation. |
| No source/citation UI | User cannot verify where the answer came from. |
| Error handling is minimal | Missing Ollama, missing model or empty PDF can crash or confuse the demo. |
| No tests/evaluation set | Hard to measure whether changes improve quality. |

## Upgrade Direction

The upgrade version should address these issues without changing the baseline reference:

- Persistent ChromaDB.
- Page/source metadata.
- Source citation display.
- Configurable models and RAG parameters.
- Error handling.
- Short chat memory.
- Notion CSV/API data source.
- Evaluation and smoke tests.
