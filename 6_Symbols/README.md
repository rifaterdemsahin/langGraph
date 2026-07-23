# 6_Symbols — Code & Implementation 💻

Concrete code snippets, runnable examples, and implementation notes that bring the concepts to life.

Purpose
- Demonstrate practical implementations that map to concepts and guides elsewhere in the structure.

What you'll find
- `run_graph_rag.py` — the runnable graph RAG PoC from [`5_Formula/formula_poc.md`](../5_Formula/formula_poc.md).
  A 3-node LangGraph workflow (`semantic_retriever` → `graph_traverser` → `generate_answer`) over the
  10 mock markdown files in `doc/`. Works out of the box with mock embeddings/LLM; set `OPENAI_API_KEY`
  for real results.
- `ingest.py` — the incremental production ingestion pipeline from [`5_Formula/formula_ingest.md`](../5_Formula/formula_ingest.md),
  writing to Neo4j (link graph) and Chroma (vectors). Requires `docker compose up -d` for those services
  (see that guide) and `OPENAI_API_KEY`.
- `doc/` — the 10 interconnected mock markdown files (with `[[wiki-links]]`) both scripts operate on.
- `requirements.txt` — dependencies for both scripts.

Quick start
```bash
cd 6_Symbols
pip install -r requirements.txt
python run_graph_rag.py "What is Deep Learning and how does it relate to NLP?"
```

Runs immediately without any API key (mock embeddings + mock LLM). Set `OPENAI_API_KEY` in your
environment first for real embeddings and a real generated answer.

How to use
- Run examples locally, inspect comments, and adapt them to your use case. Link back to `4_UI` and `5_Formula` for theory.
- See the root [`index.html`](../index.html) for a visual walkthrough of why/how/what this code does.

CLI install
- The install script will create `6_Symbols/scripts/` and place helper scripts (make executable). Use:

  bash create_learning_structure.sh

Rendering images
- Add architecture screenshots to `assets/6_symbols/` and reference in code docs.

Learning verb: Execute it — Practice by reading and running code.