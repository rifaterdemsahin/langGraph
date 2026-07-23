# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A LangChain + LangGraph proof-of-concept: graph-enhanced RAG over Obsidian-style markdown notes
linked with `[[wiki-links]]`. Plain semantic search misses notes that are related but phrased
differently; this PoC adds a second retrieval signal — following the link graph — via a LangGraph
`StateGraph`.

The repo has two layers:
1. **The code** (`6_Symbols/`) — the actual runnable Python implementation.
2. **The 8-folder learning structure** (`1_Journey/` … `8_Test/`) — a documentation scaffold where
   each folder answers a different question about the same project (see below). This structure is
   intentional and predates the code; don't restructure it without reason.

## Commands

```bash
cd 6_Symbols
pip install -r requirements.txt

# Runnable PoC — works out of the box with mock embeddings/LLM, no API key needed
python run_graph_rag.py "What is Deep Learning and how does it relate to NLP?"

# Set OPENAI_API_KEY first for real embeddings + a real generated answer
OPENAI_API_KEY=sk-... python run_graph_rag.py "your question here"

# Production ingestion pipeline (needs `docker compose up -d` for Neo4j + Chroma, see
# 5_Formula/formula_ingest.md — no docker-compose.yml is committed, it's documented inline in that guide)
OPENAI_API_KEY=sk-... python ingest.py
```

There is no test suite, linter, or build step configured in this repo.

## Architecture: the LangGraph pipeline (`6_Symbols/run_graph_rag.py`)

- **LangChain** owns document loading (`DirectoryLoader` + `TextLoader`, *not* the default
  `Unstructured` loader — that requires an extra `unstructured` package this repo doesn't depend
  on), chunking (`RecursiveCharacterTextSplitter`), embeddings, and the vector store (`Chroma`,
  in-memory).
- **A regex (`\[\[(.*?)\]\]`) extracts the `[[link]]` graph** into a plain Python dict
  (`{filename: [linked_filenames]}`) alongside the LangChain document load — this dict is the
  second retrieval signal, kept outside LangChain entirely.
- **LangGraph (`StateGraph(GraphRAGState)`)** wires exactly 3 nodes in a fixed line:
  `semantic_retriever` (vector search) → `graph_traverser` (walks the link dict from the
  semantic hits) → `generate_answer` (prompts the LLM with the union of both). State flows through
  a `TypedDict` (`GraphRAGState`) with `query`, `initial_docs`, `graph_docs`, `all_docs`,
  `final_answer`.
- **Mock fallback is load-bearing, not incidental.** If `OPENAI_API_KEY` is unset, `build_models()`
  swaps in `MockEmbeddings` (deterministic hashed bag-of-words vectors) and `MockChatModel` so the
  entire graph still runs end-to-end. Keep this working when touching `build_models()` or the node
  functions — it's what makes the PoC runnable without any external account, and what the
  `index.html` demo's "run the real script" instructions promise.

## The production path (`6_Symbols/ingest.py`)

`run_graph_rag.py` is deliberately in-memory and re-embeds everything on every run — that's the
PoC. `ingest.py` is the incremental counterpart described in `5_Formula/formula_ingest.md`: it
tracks already-processed files in `processed_files.txt` and writes new ones to Neo4j (`[[link]]`
edges as `MERGE`d graph relationships) and a Chroma HTTP server (chunks + embeddings). It requires
external services (Neo4j, Chroma server) that aren't part of this repo — see that guide for the
`docker-compose.yml` snippet. Don't assume `ingest.py` is runnable standalone the way
`run_graph_rag.py` is.

`5_Formula/formula_size.md` explains *why* this split exists (the in-memory PoC fails completely
past ~1000 files), and `5_Formula/formula_qdrant_scale.md` is the rationale for going further still
— to a 200k+-file vault — with Qdrant instead of Chroma, including the architectural choice
between a dedicated graph DB (Neo4j) vs. storing links as Qdrant payload and traversing via
payload filter.

## The 8-folder learning structure

Each top-level numbered folder has its own README explaining its purpose and "learning verb"; the
loop is: `1_Journey` (story/steps) → `2_Real` (OKRs) → `3_Environment` (roadmap/use cases) →
`4_UI` (concepts) → `5_Formula` (guides — this is where the architecture docs referenced above
live) → `6_Symbols` (code — the folder covered above) → `7_Semblance` (errors/fixes) → `8_Test`
(validation). `5_Formula/create_learning_structure.sh` is the installer that originally scaffolded
these folders and their READMEs; re-running it is idempotent (only creates missing folders/READMEs).

`index.html` at the repo root is a static, dependency-free visualization tying all of this
together — architecture diagram, an interactive client-side simulation of the exact same
10-document link graph and 3-node pipeline as `run_graph_rag.py` (word-overlap scoring instead of
real embeddings, since it has no backend), a diagram of the 8-folder learning loop, and curated
external links for the underlying concepts (LangGraph, LangChain, RAG, Qdrant, Neo4j). It's
deployed to GitHub Pages on every push to `main` via `.github/workflows/deploy-pages.yml`. If you
change the mock doc/link-graph fixtures in `6_Symbols/doc/`, update the hardcoded `DOCS`/`LINKS`
objects in `index.html`'s inline script to match, or the two will drift.
