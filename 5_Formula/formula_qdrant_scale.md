# Rationale: Scaling Graph RAG to a 200k+ file "Second Brain" with Qdrant

This extends [[formula_size.md]] and [[formula_ingest.md]] for a specific target: a personal
knowledge vault (Obsidian-style, `[[links]]` everywhere) that has grown past ~200,000 markdown
files, using **Qdrant** as the vector database instead of Chroma. It answers one question: *what
do I actually need to have in place for LangChain + LangGraph semantic search to stay fast,
affordable, and correct at this size?*

---

## 1. Why Qdrant specifically at this scale

Chroma (used in the PoC, `6_Symbols/run_graph_rag.py`) is great up to the low millions of
vectors on a single node, but three Qdrant features earn their keep once you cross ~100k source
files (which is likely 1-5M chunks after splitting):

| Need at 200k+ files | Qdrant feature |
| --- | --- |
| Filter by note metadata *before* the vector search runs (folder, tags, modified date, link count) | **Payload indexing** with pre-filtering, so a filtered search doesn't degrade to a full scan |
| Keep RAM cost sane once vectors don't fit in memory | **Scalar / product quantization** + `on_disk` payload and vectors |
| Zero-downtime re-indexing when you change embedding models | **Aliases** — build a new collection, then atomically swap the alias your app reads from |
| Horizontal growth past one machine | **Sharding + replication** built into Qdrant's distributed mode |
| Fast incremental upserts (daily new/edited notes) | Native upsert-by-id, no full collection rebuild |

## 2. The architectural fork: do you still need a separate graph database?

The PoC's `graph_traverser` node walks a `[[link]]` graph. At 200k+ files you have two honest
options:

**Option A — keep a dedicated graph DB (Neo4j/Kùzu), as in [[formula_ingest.md]].**
Best when your queries need multi-hop traversal (2-3+ hops), path-finding, or graph algorithms
(centrality, communities). Cypher is the right tool for that; Qdrant is not.

**Option B — store the link list as Qdrant payload and traverse via `payload filter`, not a
second database.**
Each chunk's payload includes `{"source_filename": "...", "links_to": ["docA.md", "docB.md"]}`.
`graph_traverser` becomes a `scroll`/`search` call with
`Filter(should=[FieldCondition(key="source_filename", match=MatchAny(any=linked_filenames))])`
instead of a Cypher query. This removes an entire moving part (no Neo4j to run/back up), at the
cost of only supporting shallow (1-2 hop) traversal efficiently.

**Recommendation:** start with Option B — it's one less system to operate, and one-hop
"what does this note link to" is what most second-brain queries actually need. Move to Option A
only once you've measured a real need for deeper multi-hop traversal.

## 3. Collection design

- **Index `source_filename`, `links_to`, `tags`, `modified_at` as payload indexes** — pre-filtering
  is what keeps a 200k-file collection fast; without it, every query degrades toward a linear scan.
- **Chunk size 500-800 tokens with ~15% overlap.** At 200k files this is roughly 1-3M points;
  test your actual average note length before committing to a number.
- **One point per chunk, not one point per file.** Store the parent filename in payload so
  `graph_traverser` and citation display can group chunks back to their source note.
- **Quantize.** Scalar quantization (int8) typically cuts memory ~4x with a small recall cost —
  worth it once the collection no longer fits comfortably in RAM. Benchmark recall on your own
  query set before shipping it.

## 4. Embeddings at this scale — the part people underestimate

- **Batch embedding calls** (e.g. 100-500 chunks per request) — at 1-3M chunks, one-at-a-time
  calls turn a same-day re-index into a multi-day job purely from HTTP round-trip overhead.
- **Cache by content hash.** Store `sha256(chunk_text)` in payload; skip re-embedding unchanged
  chunks on every re-run. This is the single biggest cost lever once your vault is mostly stable
  and only a few hundred notes change per day.
- **Rate limit + retry with backoff** against your embedding provider; a 200k-file backfill will
  hit rate limits if run naively.
- **Estimate cost before you start.** 200k files x ~6 chunks/file x $0.02/1M tokens
  (text-embedding-3-small pricing tier, illustrative) is cheap; the same math with a larger
  embedding model is not. Do the multiplication before choosing a model.

## 5. Incremental ingestion (extends [[formula_ingest.md]])

The PoC's `processed_files.txt` tracker doesn't scale past a few thousand files (linear file scan,
no way to detect *edits* to already-processed files, no concurrency safety). At 200k+ files you need:

1. **A content-hash watermark per file**, not just a processed/not-processed boolean — so edited
   notes get re-embedded, not just new ones.
2. **A real queue or database table** (SQLite is enough for a single-machine second brain) instead
   of a flat text file, so ingestion is crash-safe and restartable mid-run.
3. **Idempotent upserts keyed on a deterministic point ID** (e.g. `hash(filename + chunk_index)`),
   so re-running ingestion after a crash never creates duplicate points.
4. **Deletion handling.** When a note is deleted or renamed, its old points must be removed from
   Qdrant by `source_filename` filter — the PoC has no delete path at all.

## 6. Making the LangGraph side durable

The PoC's `StateGraph` runs start-to-finish in memory. At this scale, add:

- **A checkpointer** (LangGraph supports SQLite/Postgres-backed checkpointers) so a long-running
  `graph_traverser` step over a huge fan-out of links can resume instead of restarting from
  `semantic_retriever` on failure.
- **A `top_k` cap on graph expansion.** An unbounded traversal over a 200k-file link graph can
  fan out to hundreds of linked notes from two or three semantic hits; cap it (e.g. top 10 by
  some relevance score) before it reaches `generate_answer`, or your context window — and your
  LLM bill — blows up.
- **Structured tracing** (LangSmith, or plain structured logs) per node, so a slow or wrong answer
  can be traced back to which retrieval step went wrong, without re-running the whole graph.

## 7. Summary checklist

- [ ] Qdrant collection with payload indexes on `source_filename`, `links_to`, `tags`
- [ ] Decision made: payload-filter traversal (Option B) vs. dedicated graph DB (Option A)
- [ ] Chunking strategy sized against your actual average note length
- [ ] Quantization benchmarked (recall vs. memory) if the collection won't fit in RAM
- [ ] Batched, content-hash-cached, rate-limited embedding pipeline
- [ ] Crash-safe incremental ingestion with edit + delete handling, not a flat tracker file
- [ ] LangGraph checkpointing and a bounded `top_k` on graph expansion
- [ ] Per-node tracing so slow/wrong answers are debuggable

Related: [[formula_poc.md]] (the in-memory PoC this scales beyond), [[formula_size.md]]
(why the PoC breaks and the Chroma/Neo4j production path), [[formula_ingest.md]] (the
incremental ingestion pattern this document extends).
