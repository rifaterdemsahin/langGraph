# When It Makes Sense to Use LangChain + LangGraph

This project uses both. That's not the default right answer for every LLM project — it's the
right answer for *this* problem (graph RAG over a linked note vault). This guide separates the two
tools and lists concrete use cases where each earns its keep, and — just as importantly — where a
plain API call is the better choice.

---

## LangChain: use it when you need...

| Use case | Why LangChain fits |
| --- | --- |
| **Retrieval over your own documents (RAG)** | Loaders, text splitters, embeddings, and vector store interfaces are the exact building blocks this repo's `run_graph_rag.py` uses — writing them by hand adds nothing. |
| **Swappable models/vector stores/providers** | A common interface (`ChatOpenAI`, `Chroma`, `Qdrant`, ...) means switching providers is a constructor change, not a rewrite — see `build_models()` in `run_graph_rag.py` swapping OpenAI for mock implementations. |
| **Standard prompt templating across many call sites** | `ChatPromptTemplate` keeps prompt construction consistent and testable instead of scattered f-strings. |
| **Document loading from many formats/sources** | `DirectoryLoader`, `TextLoader`, and format-specific loaders save you from writing parsers for PDFs, HTML, markdown, etc. |

## LangChain: skip it when...

- **You're making a single, one-shot prompt call.** `client.chat.completions.create(...)` (or the
  equivalent) *is* the whole program — a framework adds a dependency tree and abstraction layer
  for something that's already three lines.
- **Your corpus is small enough to paste directly into the prompt.** No loader, splitter, or vector
  store is buying you anything if the content already fits comfortably in context.
- **You need to see and control exactly what's sent to the model**, and the abstraction is getting
  in the way of that (a common, legitimate critique of any framework layer — sometimes the
  straight-line version is more debuggable).

## LangGraph: use it when you need...

| Use case | Why LangGraph fits |
| --- | --- |
| **Multi-step retrieval with explicit intermediate state** | This repo's `graph_traverser` node depends on what `semantic_retriever` found — that's state passed through a graph, not implicit in a prompt string. |
| **Conditional branching / loops** (ReAct-style tool loops, self-correction, retries) | `StateGraph` conditional edges express "if X, go here; otherwise, loop back" as code, not as hoped-for prompt instructions. |
| **Multi-agent or multi-tool orchestration** | Supervisor/worker patterns, parallel branches that need to be joined, and tool-calling agents all need a control-flow layer above the LLM calls themselves. |
| **Resumability and durability** | A checkpointer (SQLite/Postgres-backed) lets a long-running graph survive a crash or a human-in-the-loop pause without restarting from scratch — see `formula_qdrant_scale.md` for why this matters at scale. |
| **Debuggability of *where* a pipeline went wrong** | Because each node is a named, inspectable step, you can trace a bad answer back to `semantic_retriever` returning the wrong docs vs. `generate_answer` misusing good ones — a single monolithic prompt can't be diagnosed this way. |

## LangGraph: skip it when...

- **There's no branching and no state to carry between steps.** A straight line —
  "retrieve, then generate" with nothing conditional — doesn't need a graph; a plain function call
  chain is simpler to read and debug.
- **The whole task is one LLM call.** Orchestration only pays for itself once there's more than one
  step whose outcome affects what happens next.
- **You're prototyping a one-off script that will never be re-run or extended.** The ceremony of
  defining state, nodes, and edges is worth it for something that will grow; it's overhead for a
  script you'll throw away tomorrow.

## Quick decision checklist

- [ ] Do I need to retrieve from documents I don't want to paste into every prompt? → **LangChain**
- [ ] Do I need more than one LLM/tool call where a later step depends on an earlier one's result? → **LangGraph**
- [ ] Is there real branching, looping, or multi-agent coordination? → **LangGraph**
- [ ] Do I need to resume a long-running workflow after a crash or a pause for human approval? → **LangGraph**
- [ ] Is this genuinely a single prompt → single response, over content that already fits in context? → **Neither — just call the model directly.**

Related: [[formula_poc.md]] (the concrete LangGraph pipeline this guide is drawn from),
[[formula_qdrant_scale.md]] (why orchestration matters *more*, not less, as scale grows — the
"do we still need this with a bigger context window" question).
