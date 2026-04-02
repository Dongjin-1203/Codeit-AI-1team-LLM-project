# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RFPilot** — A B2G (Business-to-Government) chatbot that answers questions about Korean public procurement RFP (제안요청서) documents using RAG (Retrieval-Augmented Generation). The system uses OpenAI GPT models or a local quantized Llama GGUF model (Llama-3-Open-Ko-8B).

## Setup

**Dependencies managed with Poetry (Python 3.12.3 required):**
```bash
python -m poetry config virtualenvs.in-project true
python -m poetry env use 3.12.3
python -m poetry install
python -m poetry shell
```

**Environment variables** — copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY` — required for GPT models and OpenAI embeddings
- `LANGCHAIN_TRACING_V2`, `LANGSMITH_API_KEY`, `LANGCHAIN_PROJECT` — optional, for experiment tracking
- `WANDB_API_KEY` — optional, for W&B tracking
- `USE_MODEL_HUB` — `true` (default) downloads GGUF from HuggingFace Hub; `false` uses local `GGUF_MODEL_PATH`

## Common Commands

```bash
# Run full pipeline (preprocess → embed → vectordb)
python main.py --step all

# Individual pipeline steps
python main.py --step preprocess
python main.py --step embed
python main.py --step rag --query "질문 입력"

# Launch chatbot UI
streamlit run src/visualization/chatbot_app.py

# Run evaluation experiments (interactive menu)
python src/evaluation/run_experiment.py
python src/evaluation/run_experiment.py --run      # run experiment
python src/evaluation/run_experiment.py --compare  # compare results

# Linting / formatting (dev dependencies)
black .
pytest
```

## Architecture

The RAG pipeline has 5 main stages, each in its own `src/` module:

### 1. `src/loader/` — Preprocessing
`RAGPreprocessPipeline` reads PDF/HWP files from `data/files/`, extracts text, and chunks it (default: 1000 chars, 200 overlap) using LangChain text splitters. Output is saved as `data/rag_chunks_final.csv`.

### 2. `src/embedding/` — Vector DB Construction
`RAGVectorDBPipeline` reads the chunks CSV, creates embeddings via OpenAI `text-embedding-3-small`, and persists to ChromaDB at `./chroma_db/` (collection name: `rag_documents`).

### 3. `src/retriever/` — Document Search
`RAGRetriever` supports three search modes:
- `embedding` — pure vector similarity via ChromaDB
- `hybrid` — BM25 (`rank-bm25`) + embedding scores combined with a tunable `alpha` weight
- `hybrid_rerank` (default) — hybrid search followed by `BAAI/bge-reranker-base` CrossEncoder re-ranking

### 4. `src/router/` — Query Classification
`QueryRouter.classify()` uses keyword matching to route queries into four types: `greeting`, `thanks`, `document`, `out_of_scope`. Only `document` queries go through RAG retrieval; others get hardcoded direct responses.

### 5. `src/generator/` — Answer Generation
`RAGPipeline` orchestrates the full chain: router → retriever → LangChain `ChatOpenAI` prompt → answer. Maintains multi-turn `chat_history`. Default model: `gpt-4o-mini`. For local inference, `generator_gguf.py` uses `llama-cpp-python`.

### Configuration
`src/utils/config.py` has a single `Config` class (aliased as `PreprocessConfig` and `RAGConfig`) that loads all settings from environment variables. All modules accept a `config` parameter and default to creating a new `Config()` instance.

### UI
`src/visualization/chatbot_app.py` is a Streamlit app. The user provides their OpenAI API key in the sidebar, selects a model, and configures search settings. Uses `src/utils/conversation_manager.py` for session state.

### Experiment Tracking
`src/evaluation/` uses LangSmith for tracing (via `@traceable` decorators on `RAGRetriever` and `RAGPipeline` methods) and RAGAS metrics for evaluation. W&B is optionally used for logging.

## Key Data Files

- `data/rag_chunks_final.csv` — preprocessed RAG chunks (tracked with Git LFS)
- `chroma_db/` — persisted ChromaDB vector store (tracked with Git LFS)
- `data/files/` — original RFP documents (not in repo; download separately)
