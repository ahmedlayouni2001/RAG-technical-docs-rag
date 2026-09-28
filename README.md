# Technical Docs RAG Agent

[Live demo (frontend)](https://rag-technical-wv5.vercel.app/) · [Source](https://github.com/ahmedlayouni2001/RAG-technical-docs-rag)

A Corrective RAG system for querying technical documentation — hybrid retrieval
(BM25 + vector search) feeding a self-correcting LangGraph agent that grades
its own retrieval and retries with a rewritten query before it ever answers.

Built against FastAPI's own documentation (155 real markdown files) as a
concrete example. Swap `data/docs` for any other markdown-based docs.

## Why this isn't just another RAG tutorial

- **Hybrid retrieval, not just vector search.** BM25 (exact keyword matches —
  function names, error codes) fused with semantic vector search via
  Reciprocal Rank Fusion. Pure vector search misses exact terms; pure
  keyword search misses paraphrasing. This uses both.
- **Self-correcting, not a fixed pipeline.** A LangGraph agent grades whether
  retrieved chunks are actually relevant. If not, it rewrites the query and
  retries (capped, so it can't loop forever) instead of confidently
  answering from irrelevant context.
- **Structure-aware ingestion.** Chunks split on markdown headers first (so
  a chunk never straddles two unrelated sections), each stamped with a
  contextual header (file + heading breadcrumb) so it still makes sense once
  it's floating alone in the vector store.

## Architecture

```
 User → Frontend → Backend (FastAPI + LangGraph agent)
                          │
                          ├─→ Hybrid retriever (BM25 + vector, RRF-fused)
                          └─→ LLM API (generation + relevance grading)
```

Agent loop: `retrieve → grade_documents → (generate | transform_query → retrieve)`

## Deployment status

- **Frontend** — live on Vercel: https://rag-technical-wv5.vercel.app/
  (static `frontend/index.html`, deployed straight from this repo via
  `vercel.json`).
- **Backend** — not deployed yet. The FastAPI + LangGraph API persists a
  Chroma vector store to disk and embeds lazily on first request, which
  doesn't fit Vercel's stateless serverless functions well. It needs a host
  with persistent disk and a long-running process (Railway, Render, Fly.io).
  Until then, the live demo's chat UI loads but `/health` and `/chat` calls
  will 404 — run the backend locally (see Setup) to try the full agent.

## Status

- [x] Phase 1 — Ingestion (header-aware chunking, contextual headers)
- [x] Phase 2 — Hybrid retrieval (BM25 + vector, EnsembleRetriever)
- [x] Phase 3 — Agent loop (LangGraph corrective RAG)
- [x] Phase 4 — Evaluation harness (RAGAS golden-set scoring)
- [x] Phase 5 — Backend API (FastAPI, streaming)
- [x] Phase 6 — Frontend (chat UI with source highlighting)
- [ ] Phase 7 — Deployment (frontend live on Vercel; backend hosting still pending)
- [ ] Phase 8 — Reranking, evaluation numbers in this README

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY

# Test ingestion (no API key needed — pure text processing)
python -m src.ingestion.loader

# Test hybrid retrieval (requires OPENAI_API_KEY — embeds on first run)
python -m src.retrieval.hybrid_retriever

# Test the full agent loop
python -m src.agent.graph

# Run the API (serves the frontend too, at http://localhost:8000)
uvicorn src.api.main:app --reload
```

## Project layout

```
src/
  ingestion/   Markdown loading + header-aware chunking
  retrieval/   BM25 + vector hybrid retriever
  agent/       LangGraph corrective RAG loop
  api/         FastAPI backend
  config.py    All tunable settings in one place
data/docs/     Source documentation (currently: FastAPI's docs)
evaluation/    RAGAS golden-set evaluation
frontend/      Chat UI (static, deployed to Vercel)
```
