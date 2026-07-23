#!/usr/bin/env python3
"""
Graph RAG proof-of-concept: LangChain for loading/embedding/retrieval,
LangGraph for orchestrating a stateful, multi-step retrieval workflow.

Pipeline:
  1. Document Loading (LangChain) - load markdown files from ./doc
  2. Graph Extraction        - parse [[wiki-links]] into an in-memory link graph
  3. Vector Store (LangChain) - embed + store docs for semantic search
  4. Graph Definition (LangGraph) - a 3-node stateful workflow:
       semantic_retriever -> graph_traverser -> generate_answer

Run without any API key and it falls back to deterministic mock
embeddings/LLM so the whole graph is runnable out of the box. Set
OPENAI_API_KEY to get real embeddings and real answers.

Usage:
    python run_graph_rag.py "What is Deep Learning and how does it relate to NLP?"
"""

import os
import re
import sys
from typing import Dict, List, Tuple, TypedDict

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

DOC_DIR = os.path.join(os.path.dirname(__file__), "doc")

# --- Embeddings / LLM: real if OPENAI_API_KEY is set, mock otherwise ---


class MockEmbeddings:
    """Deterministic bag-of-words vectors so mock semantic search is stable."""

    DIM = 64

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.DIM
        for word in re.findall(r"[a-zA-Z]+", text.lower()):
            vec[hash(word) % self.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class MockChatModel:
    def invoke(self, prompt) -> str:
        return (
            "This is a mock answer generated without a real LLM. "
            "Set OPENAI_API_KEY to get a genuine response from the "
            "retrieved context above."
        )


def build_models() -> Tuple[object, object]:
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        return OpenAIEmbeddings(), ChatOpenAI(model="gpt-4o", temperature=0)
    print("WARNING: OPENAI_API_KEY not set. Using mock embeddings/LLM.\n")
    return MockEmbeddings(), MockChatModel()


# --- 1 & 2. Document loading + [[link]] graph extraction ---

LINK_PATTERN = re.compile(r"\[\[(.*?)\]\]")


def load_and_parse_docs(directory: str) -> Tuple[List[Document], Dict[str, List[str]]]:
    loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

    graph: Dict[str, List[str]] = {}
    for doc in docs:
        source_filename = os.path.basename(doc.metadata.get("source", "unknown"))
        graph.setdefault(source_filename, [])
        for link in LINK_PATTERN.findall(doc.page_content):
            if not link.endswith(".md"):
                link += ".md"
            if link not in graph[source_filename]:
                graph[source_filename].append(link)
    return docs, graph


# --- 3. Vector store ---


def create_vector_store(docs: List[Document], embeddings) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    return Chroma.from_documents(documents=splits, embedding=embeddings)


# --- 4. LangGraph state ---


class GraphRAGState(TypedDict):
    query: str
    initial_docs: List[Document]
    graph_docs: List[Document]
    all_docs: List[Document]
    final_answer: str


# --- 5. Nodes ---


def make_nodes(document_store, link_graph, vector_store, llm):
    def semantic_retriever(state: GraphRAGState) -> GraphRAGState:
        print("--- Node: semantic_retriever ---")
        query = state["query"]
        retrieved = vector_store.similarity_search(query, k=2)
        print(f"Found {len(retrieved)} semantic docs.")
        return {**state, "initial_docs": retrieved}

    def graph_traverser(state: GraphRAGState) -> GraphRAGState:
        print("--- Node: graph_traverser ---")
        initial_docs = state["initial_docs"]
        all_by_name = {
            os.path.basename(d.metadata.get("source")): d for d in document_store
        }

        graph_docs = []
        for doc in initial_docs:
            source_name = os.path.basename(doc.metadata.get("source"))
            linked_names = link_graph.get(source_name, [])
            print(f"Doc '{source_name}' has links to: {linked_names}")
            for name in linked_names:
                if name in all_by_name:
                    graph_docs.append(all_by_name[name])

        print(f"Found {len(graph_docs)} graph-linked docs.")
        combined = {d.metadata["source"]: d for d in initial_docs + graph_docs}
        return {**state, "graph_docs": graph_docs, "all_docs": list(combined.values())}

    def generate_answer(state: GraphRAGState) -> GraphRAGState:
        print("--- Node: generate_answer ---")
        query, all_docs = state["query"], state["all_docs"]
        if not all_docs:
            return {**state, "final_answer": "Sorry, I couldn't find any relevant information."}

        prompt_template = """
        You are an assistant for question-answering tasks. Use the following pieces
        of retrieved context to answer the question. If you don't know the answer,
        say so. Use three sentences maximum and keep the answer concise.

        Question: {question}

        Context:
        {context}

        Answer:
        """
        context_str = "\n\n---\n\n".join(
            f"Source: {os.path.basename(d.metadata.get('source'))}\n\n{d.page_content}"
            for d in all_docs
        )
        prompt = ChatPromptTemplate.from_template(prompt_template).invoke(
            {"question": query, "context": context_str}
        )
        answer = llm.invoke(prompt)
        if hasattr(answer, "content"):
            answer = answer.content
        return {**state, "final_answer": answer}

    return semantic_retriever, graph_traverser, generate_answer


def build_graph(semantic_retriever, graph_traverser, generate_answer):
    workflow = StateGraph(GraphRAGState)
    workflow.add_node("semantic_retriever", semantic_retriever)
    workflow.add_node("graph_traverser", graph_traverser)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("semantic_retriever")
    workflow.add_edge("semantic_retriever", "graph_traverser")
    workflow.add_edge("graph_traverser", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def main() -> None:
    if not os.path.exists(DOC_DIR):
        print(f"Error: '{DOC_DIR}' not found.")
        sys.exit(1)

    query = " ".join(sys.argv[1:]) or "What is Deep Learning and how does it relate to NLP?"

    embeddings, llm = build_models()

    print(f"Loading documents from {DOC_DIR}...")
    document_store, link_graph = load_and_parse_docs(DOC_DIR)

    print("--- Extracted Link Graph ---")
    for k, v in link_graph.items():
        if v:
            print(f"{k} -> {v}")
    print("------------------------------")

    print("Splitting and embedding documents...")
    vector_store = create_vector_store(document_store, embeddings)

    semantic_retriever, graph_traverser, generate_answer = make_nodes(
        document_store, link_graph, vector_store, llm
    )
    app = build_graph(semantic_retriever, graph_traverser, generate_answer)

    print("\n" + "=" * 50)
    print(f"Running Graph RAG for query: '{query}'")
    print("=" * 50 + "\n")

    final_state = app.invoke({"query": query})

    print("\n" + "=" * 50)
    print("--- Final Result ---")
    print(f"Query: {query}")
    print(f"\nFinal Answer: {final_state['final_answer']}")

    print("\n--- Documents Used ---")
    for doc in final_state["all_docs"]:
        print(f"- {os.path.basename(doc.metadata.get('source'))}")
    print("=" * 50)


if __name__ == "__main__":
    main()
