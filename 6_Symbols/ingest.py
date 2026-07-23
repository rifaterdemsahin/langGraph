#!/usr/bin/env python3
"""
Production ingestion pipeline (see 5_Formula/formula_ingest.md).

Where run_graph_rag.py is an in-memory PoC that re-embeds everything on
every run, this script is the incremental, persistent counterpart: it
tracks which files have already been ingested and only processes new
ones, writing:
  - vectors + text   -> Chroma (running as a server, see docker-compose.yml)
  - [[link]] edges   -> Neo4j (running as a server, see docker-compose.yml)

Requires: docker compose up -d  (see 6_Symbols/docker-compose.yml)
          OPENAI_API_KEY set in the environment

Usage:
    python ingest.py
"""

import os
import re

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "your-strong-password")

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = os.environ.get("CHROMA_PORT", "8000")
CHROMA_COLLECTION = "markdown_docs"

DOC_FOLDER = os.path.join(os.path.dirname(__file__), "doc")
TRACKER_FILE = os.path.join(os.path.dirname(__file__), "processed_files.txt")

LINK_PATTERN = re.compile(r"\[\[(.*?)\]\]")


def get_embedding_function():
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
    )


def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Neo4j connection successful.")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None


def get_chroma_collection():
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
        client.heartbeat()
        print("Chroma connection successful.")
        return client.get_or_create_collection(
            name=CHROMA_COLLECTION, embedding_function=get_embedding_function()
        )
    except Exception as e:
        print(f"Failed to connect to Chroma: {e}")
        return None


def process_new_file(filepath, neo4j_driver, chroma_collection) -> bool:
    filename = os.path.basename(filepath)
    print(f"--- Processing: {filename} ---")
    try:
        doc = TextLoader(filepath).load()[0]
        content = doc.page_content
        links = LINK_PATTERN.findall(content)

        with neo4j_driver.session() as session:
            session.run("MERGE (d:Document {name: $filename})", filename=filename)
            for link in links:
                target_name = link if link.endswith(".md") else f"{link}.md"
                session.run("MERGE (t:Document {name: $target_name})", target_name=target_name)
                session.run(
                    """
                    MATCH (d:Document {name: $source_name})
                    MATCH (t:Document {name: $target_name})
                    MERGE (d)-[:LINKS_TO]->(t)
                    """,
                    source_name=filename,
                    target_name=target_name,
                )
            print(f"Neo4j: added {filename} with {len(links)} link(s).")

        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents([doc])
        if splits:
            chroma_collection.add(
                documents=[s.page_content for s in splits],
                metadatas=[{"source_filename": filename, "chunk_index": i} for i in range(len(splits))],
                ids=[f"{filename}_{i}" for i in range(len(splits))],
            )
            print(f"Chroma: added {len(splits)} chunk(s) for {filename}.")
        return True
    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        return False


def load_processed_files(path) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f}


def save_processed_file(path, filename) -> None:
    with open(path, "a") as f:
        f.write(f"{filename}\n")


def run_ingestion() -> None:
    print("Starting ingestion pipeline...")
    neo4j_driver = get_neo4j_driver()
    chroma_collection = get_chroma_collection()
    if not neo4j_driver or not chroma_collection:
        print("Failed to connect to databases. Exiting.")
        return

    processed = load_processed_files(TRACKER_FILE)
    new_files = [f for f in os.listdir(DOC_FOLDER) if f.endswith(".md") and f not in processed]

    if not new_files:
        print("No new files to process.")
        return

    print(f"Found {len(new_files)} new file(s): {new_files}")
    success = 0
    for filename in new_files:
        if process_new_file(os.path.join(DOC_FOLDER, filename), neo4j_driver, chroma_collection):
            save_processed_file(TRACKER_FILE, filename)
            success += 1

    print(f"\nIngestion complete. Processed {success} new file(s).")
    neo4j_driver.close()


if __name__ == "__main__":
    if not os.path.exists(DOC_FOLDER):
        print(f"Error: '{DOC_FOLDER}' not found.")
    else:
        run_ingestion()
