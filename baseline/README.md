# Multilingual PDF RAG Chatbot (Cohere + Gemini)

This folder contains a Streamlit PDF RAG Chatbot implementation that leverages **Cohere** for high-quality multilingual document embeddings and **Gemini** for response generation.

## Features

- **Document Ingestion:** Upload PDF files and extract text using `pypdf`.
- **Vector Embeddings:** Uses Cohere's `embed-multilingual-v3.0` API to convert document chunks and user queries into dense vectors (1024 dimensions), ensuring excellent retrieval quality for Vietnamese and other languages.
- **Session Storage:** Uses in-memory `chromadb` to store and retrieve vectors locally for the duration of the Streamlit session.
- **Answer Generation:** Uses Google's Gemini (e.g., `gemini-2.5-flash`) to generate accurate, context-aware answers based on the retrieved context.

## Setup & Deployment

### 1. Requirements
Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

### 2. API Keys & Secrets if you want to deploy on your own
**Do not hardcode API keys in the source code.**
Create a file named `.streamlit/secrets.toml` in this folder (or configure **Advanced settings > Secrets** in Streamlit Community Cloud) with your keys:

```toml
COHERE_API_KEY = "your_cohere_key_here"
GEMINI_API_KEY = "your_gemini_key_here"
```

### 3. Run Locally

Navigate to this folder and run:
```bash
streamlit run chatbot_app_cohere.py
```
