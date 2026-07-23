# langGraph

A proof-of-concept graph-enhanced RAG (Retrieval-Augmented Generation) system built with
**LangChain** and **LangGraph**, over a set of Obsidian-style markdown notes linked with
`[[wiki-links]]`.

**[Open the visual walkthrough (`index.html`)](index.html)** — why this exists, how the pipeline
is wired as a LangGraph state machine, an interactive client-side demo of the retrieval graph, and
links to learn every underlying concept.

## Quick start

```bash
cd 6_Symbols
pip install -r requirements.txt
python run_graph_rag.py "What is Deep Learning and how does it relate to NLP?"
```

Runs immediately with mock embeddings/LLM — no API key required to see the full pipeline execute.
Set `OPENAI_API_KEY` for real embeddings and a real generated answer.

## What's here

- **`6_Symbols/run_graph_rag.py`** — the runnable PoC: LangChain loads and embeds 10 mock markdown
  files, LangGraph orchestrates a 3-node workflow (`semantic_retriever` → `graph_traverser` →
  `generate_answer`) that combines vector search with `[[link]]`-graph traversal.
- **`6_Symbols/ingest.py`** — the production-scale counterpart: incremental ingestion into Neo4j
  (link graph) + Chroma (vectors), for when the in-memory PoC stops scaling.
- **`index.html`** — a static, self-contained visualization of the whole project: architecture
  diagram, an interactive demo of the retrieval graph, an 8-folder learning-path diagram, and
  curated links to learn LangChain/LangGraph/RAG/Qdrant/Neo4j concepts.

## Learning structure

This repo is organized as an 8-folder learning loop, each folder answering a different question
about the same project:

| Folder | Question it answers |
| --- | --- |
| [`1_Journey`](1_Journey/) | What's the story and the step-by-step path? |
| [`2_Real`](2_Real/) | What are the objectives and key results? |
| [`3_Environment`](3_Environment/) | What's the roadmap and real-world use case? |
| [`4_UI`](4_UI/) | What are the underlying concepts? |
| [`5_Formula`](5_Formula/) | What are the step-by-step guides? |
| [`6_Symbols`](6_Symbols/) | What does the running code look like? |
| [`7_Semblance`](7_Semblance/) | What broke, and how was it fixed? |
| [`8_Test`](8_Test/) | How is it validated? |

Key guides in `5_Formula/`:
- [`formula_poc.md`](5_Formula/formula_poc.md) — the PoC this repo implements
- [`formula_size.md`](5_Formula/formula_size.md) — why the in-memory PoC breaks at scale
- [`formula_ingest.md`](5_Formula/formula_ingest.md) — the incremental ingestion pipeline
- [`formula_qdrant_scale.md`](5_Formula/formula_qdrant_scale.md) — what a 200k+ file "second brain"
  needs from Qdrant + LangGraph specifically

## Deployment

`index.html` is a static, dependency-free page — it's deployed via GitHub Pages straight from the
repo root on every push to `main` (see `.github/workflows/deploy-pages.yml`).
