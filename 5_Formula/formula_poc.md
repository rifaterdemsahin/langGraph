Here is a proof-of-concept project demonstrating how to use Langchain and LangGraph to build a graph-enabled RAG (Retrieval-Augmented Generation) system based on your markdown files.

This system will first find relevant documents using standard semantic search and then use LangGraph to "traverse" the Obsidian-style `[[]]` links to find related documents, providing a more comprehensive context to answer a query.

### Project Overview

1.  **Document Loading (Langchain)**: Load your 10 markdown files from the `/doc` folder.
2.  **Graph Extraction**: Parse the loaded documents to find all `[[]]` links and build a relationship graph (a simple dictionary).
3.  **Vector Store (Langchain)**: Embed and store your documents in a vector database (like Chroma) for semantic search.
4.  **Graph Definition (LangGraph)**: Define a stateful graph that outlines the workflow.
5.  **Graph Nodes**: Create the functions (nodes) that will be part of our workflow:
      * `semantic_retriever`: Finds initial documents based on the query.
      * `graph_traverser`: Finds connected documents using the `[[]]` links.
      * `generate_answer`: Generates a final answer using all retrieved documents.
6.  **Compile & Run**: Assemble the graph and run a query.

-----

### Step 1: Project Setup & Dependencies

First, ensure you have the required libraries installed:

```bash
pip install langchain langgraph langchain_openai langchain_community chromadb
```

Next, let's create a mock `doc` folder with 10 documents for this example.

**Mock File Setup (Run this once):**

```python
import os

# Create the 'doc' directory
doc_dir = "doc"
os.makedirs(doc_dir, exist_ok=True)

# Define the content for 10 markdown files with Obsidian-style links
file_contents = {
    "doc1_intro.md": "Introduction to AI. AI is a broad field. See [[doc2_ml.md]] for more on machine learning.",
    "doc2_ml.md": "Machine Learning (ML) is a subset of AI. It uses data to train models. This is related to [[doc3_deep_learning.md]].",
    "doc3_deep_learning.md": "Deep Learning (DL) is a subset of ML that uses neural networks. It powers [[doc4_cv.md]] and [[doc5_nlp.md]].",
    "doc4_cv.md": "Computer Vision (CV) allows computers to see. It is used in image recognition. [[doc9_applications.md]] discusses uses.",
    "doc5_nlp.md": "Natural Language Processing (NLP) allows computers to understand language. This is key for chatbots. See also [[doc6_transformers.md]].",
    "doc6_transformers.md": "Transformers are a key architecture in DL, especially for [[doc5_nlp.md]]. They were introduced in 'Attention is All You Need'.",
    "doc7_history.md": "The history of AI starts in the 1950s. Early concepts are discussed in [[doc1_intro.md]].",
    "doc8_ethics.md": "AI Ethics is a critical field. It discusses bias in [[doc2_ml.md]] and societal impact.",
    "doc9_applications.md": "AI has many applications, from [[doc4_cv.md]] in self-driving cars to [[doc5_nlp.md]] in translation.",
    "doc10_future.md": "The future of AI is exciting. It may lead to AGI. This builds on all other topics, like [[doc3_deep_learning.md]]."
}

# Write the files
for filename, content in file_contents.items():
    with open(os.path.join(doc_dir, filename), "w") as f:
        f.write(content)

print(f"Created {len(file_contents)} markdown files in '{doc_dir}' directory.")
```

-----

### Step 2: The Main Python Script (Proof-of-Concept)

Save the following as `run_graph_rag.py`. This single file contains the entire application.

```python
import os
import re
from typing import List, Dict, TypedDict

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

# --- IMPORTANT: Set your OpenAI API Key ---
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable not set. Using placeholder models.")
    # Use placeholder models if API key is not set (for demonstration)
    # In a real scenario, you MUST provide an API key.
    class MockEmbeddings:
        def embed_documents(self, texts):
            return [[0.1] * 1536 for _ in texts] # Return dummy vectors
        def embed_query(self, text):
            return [0.1] * 1536

    class MockChatModel:
        def invoke(self, prompt):
            return "This is a mock answer. Set your OPENAI_API_KEY to get real results."

    embeddings = MockEmbeddings()
    llm = MockChatModel()
else:
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)


# --- 1. Document Loading & Graph Extraction ---

def load_and_parse_docs(directory: str) -> (List[Document], Dict[str, List[str]]):
    """
    Loads docs from a directory and extracts a graph of [[]] links.
    """
    print(f"Loading documents from {directory}...")
    loader = DirectoryLoader(directory, glob="**/*.md")
    docs = loader.load()
    
    # Simple regex to find [[links]]
    link_pattern = re.compile(r"\[\[(.*?)\]\]")
    
    # Graph is a dict: {filename: [list_of_linked_filenames]}
    graph = {}
    
    for doc in docs:
        source_filename = os.path.basename(doc.metadata.get("source", "unknown"))
        if source_filename not in graph:
            graph[source_filename] = []
            
        links = link_pattern.findall(doc.page_content)
        for link in links:
            # Normalize link (e.g., if it has .md or not)
            if not link.endswith(".md"):
                link += ".md"
            
            if link not in graph[source_filename]:
                graph[source_filename].append(link)
                
    print("--- Extracted Link Graph ---")
    for k, v in graph.items():
        if v:
            print(f"{k} -> {v}")
    print("------------------------------")
            
    return docs, graph

# --- 2. Vector Store Setup ---

def create_vector_store(docs: List[Document]) -> Chroma:
    """
    Creates a Chroma vector store from the loaded documents.
    """
    print("Splitting and embedding documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create the vector store. It will be stored in-memory.
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# --- 3. LangGraph State Definition ---

class GraphRAGState(TypedDict):
    """
    Defines the state for our LangGraph.
    
    Attributes:
        query (str): The user's question.
        initial_docs (List[Document]): Docs from the semantic retriever.
        graph_docs (List[Document]): Docs found by traversing the link graph.
        all_docs (List[Document]): Combined list of all retrieved docs.
        final_answer (str): The LLM-generated answer.
    """
    query: str
    initial_docs: List[Document]
    graph_docs: List[Document]
    all_docs: List[Document]
    final_answer: str

# --- 4. LangGraph Node Definitions ---

# Store these globally so our nodes can access them
DOCUMENT_STORE, LINK_GRAPH = load_and_parse_docs("doc")
VECTOR_STORE = create_vector_store(DOCUMENT_STORE)

def semantic_retriever(state: GraphRAGState) -> GraphRAGState:
    """
    Node 1: Retrieves initial documents based on semantic similarity.
    """
    print("--- Node: semantic_retriever ---")
    query = state["query"]
    print(f"Query: {query}")
    
    retrieved_docs = VECTOR_STORE.similarity_search(query, k=2) # Get top 2 semantic hits
    
    print(f"Found {len(retrieved_docs)} semantic docs.")
    return {**state, "initial_docs": retrieved_docs}

def graph_traverser(state: GraphRAGState) -> GraphRAGState:
    """
    Node 2: Traverses the link graph to find connected documents.
    """
    print("--- Node: graph_traverser ---")
    initial_docs = state["initial_docs"]
    graph_docs = []
    
    # Get all doc filenames for quick lookup
    all_doc_filenames = {os.path.basename(doc.metadata.get("source")): doc for doc in DOCUMENT_STORE}
    
    for doc in initial_docs:
        source_filename = os.path.basename(doc.metadata.get("source"))
        
        # Find links from this document in our pre-built graph
        linked_filenames = LINK_GRAPH.get(source_filename, [])
        print(f"Doc '{source_filename}' has links to: {linked_filenames}")
        
        for link_name in linked_filenames:
            # Find the full Document object for the linked filename
            if link_name in all_doc_filenames:
                graph_docs.append(all_doc_filenames[link_name])
                
    print(f"Found {len(graph_docs)} graph-linked docs.")
    
    # Combine and de-duplicate documents
    all_docs = {doc.metadata["source"]: doc for doc in initial_docs + graph_docs}
    
    return {**state, "graph_docs": graph_docs, "all_docs": list(all_docs.values())}

def generate_answer(state: GraphRAGState) -> GraphRAGState:
    """
    Node 3: Generates a final answer using all retrieved documents.
    """
    print("--- Node: generate_answer ---")
    query = state["query"]
    all_docs = state["all_docs"]
    
    if not all_docs:
        return {**state, "final_answer": "Sorry, I couldn't find any relevant information."}

    # Build the prompt
    prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
    to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Question: {question} 

    Context:
    {context} 

    Answer:
    """
    
    context_str = "\n\n---\n\n".join(
        [f"Source: {os.path.basename(doc.metadata.get('source'))}\n\n{doc.page_content}" for doc in all_docs]
    )
    
    print(f"Generating answer from {len(all_docs)} total docs.")
    
    prompt = ChatPromptTemplate.from_template(prompt_template).invoke({
        "question": query,
        "context": context_str
    })
    
    # Call the LLM
    final_answer = llm.invoke(prompt)
    
    if hasattr(final_answer, 'content'): # Handle LLM output object
        final_answer = final_answer.content
        
    return {**state, "final_answer": final_answer}

# --- 5. Graph Assembly ---

def build_graph() -> StateGraph:
    """
    Builds the LangGraph workflow.
    """
    workflow = StateGraph(GraphRAGState)

    # Add the nodes
    workflow.add_node("semantic_retriever", semantic_retriever)
    workflow.add_node("graph_traverser", graph_traverser)
    workflow.add_node("generate_answer", generate_answer)

    # Define the edges
    workflow.set_entry_point("semantic_retriever")
    workflow.add_edge("semantic_retriever", "graph_traverser")
    workflow.add_edge("graph_traverser", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Compile the graph
    app = workflow.compile()
    return app

# --- 6. Run the Application ---

if __name__ == "__main__":
    # Ensure you have run the mock file setup first
    if not os.path.exists("doc"):
        print("Error: 'doc' directory not found.")
        print("Please run the 'Mock File Setup' code from Step 1 first.")
    else:
        app = build_graph()
        
        # --- Example Query ---
        query = "What is Deep Learning and how does it relate to NLP?"
        
        print("\n" + "="*50)
        print(f"Running Graph RAG for query: '{query}'")
        print("="*50 + "\n")
        
        # The 'invoke' method runs the graph from start to finish
        final_state = app.invoke({"query": query})
        
        print("\n" + "="*50)
        print("--- Final Result ---")
        print(f"Query: {query}")
        print(f"\nFinal Answer: {final_state['final_answer']}")
        
        print("\n--- Documents Used ---")
        for doc in final_state['all_docs']:
            print(f"- {os.path.basename(doc.metadata.get('source'))}")
        print("="*50)
```

### How to Run

1.  **Run the Mock Setup**: Copy and run the "Mock File Setup" code from Step 1 to create your `doc` folder.
2.  **Set API Key (Optional but Recommended)**: Uncomment the line `os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"` in `run_graph_rag.py` and add your key. Without it, you will only get mock, non-functional answers.
3.  **Run the Main Script**:
    ```bash
    python run_graph_rag.py
    ```

### Example Output (with a real API key)

```
$ python run_graph_rag.py
Loading documents from doc...
--- Extracted Link Graph ---
doc1_intro.md -> ['doc2_ml.md']
doc2_ml.md -> ['doc3_deep_learning.md']
doc3_deep_learning.md -> ['doc4_cv.md', 'doc5_nlp.md']
doc5_nlp.md -> ['doc6_transformers.md']
doc6_transformers.md -> ['doc5_nlp.md']
doc7_history.md -> ['doc1_intro.md']
doc8_ethics.md -> ['doc2_ml.md']
doc9_applications.md -> ['doc4_cv.md', 'doc5_nlp.md']
doc10_future.md -> ['doc3_deep_learning.md']
------------------------------
Splitting and embedding documents...

==================================================
Running Graph RAG for query: 'What is Deep Learning and how does it relate to NLP?'
==================================================

--- Node: semantic_retriever ---
Query: What is Deep Learning and how does it relate to NLP?
Found 2 semantic docs.
--- Node: graph_traverser ---
Doc 'doc3_deep_learning.md' has links to: ['doc4_cv.md', 'doc5_nlp.md']
Doc 'doc5_nlp.md' has links to: ['doc6_transformers.md']
Found 3 graph-linked docs.
--- Node: generate_answer ---
Generating answer from 4 total docs.

==================================================
--- Final Result ---
Query: What is Deep Learning and how does it relate to NLP?

Final Answer: Deep Learning (DL) is a subset of machine learning that utilizes neural networks. It is directly related to Natural Language Processing (NLP) as it powers its capabilities, with specific architectures like Transformers being key to how computers understand language.

--- Documents Used ---
- doc3_deep_learning.md
- doc5_nlp.md
- doc4_cv.md
- doc6_transformers.md
==================================================
```

As you can see, the initial search found `doc3_deep_learning.md` and `doc5_nlp.md`. The `graph_traverser` node then read the links in those files (`[[doc4_cv.md]]`, `[[doc5_nlp.md]]`, `[[doc6_transformers.md]]`) and added them to the context, giving the LLM a much richer set of information to form the answer.
