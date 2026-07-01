# Smoke Tests - AIO-S1-010

## Purpose

These checks validate that Sprint 1 structure is usable before feature work starts.

## Commands

Run from the repository root.

### Compile baseline and upgrade

```bash
python -m compileall baseline upgrade
```

Expected result: Python files compile without syntax errors.

### Check upgrade imports

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Expected result: smoke tests pass.

### Run baseline manually

```bash
streamlit run baseline/chatbot_app_native.py
```

Expected result: Streamlit opens the original PDF RAG Chatbot. This requires local dependencies and Ollama models.

### Run upgrade placeholder

```bash
python upgrade/app.py
```

Expected result: command prints that the upgrade skeleton is ready.

