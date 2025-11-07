Here is a detailed, step-by-step example of an ingestion pipeline that addresses your scaling concerns.

This pipeline is designed to be **incremental** and **persistent**. It will:

1.  Use **Docker** to easily run persistent, server-based versions of **Neo4j** (Graph DB) and **Chroma** (Vector DB).
2.  Process a directory of markdown files.
3.  Keep track of files it has already processed so it only adds *new* ones.
4.  For each new file, it will:
    a.  Add the file and its `[[]]` links as nodes and relationships in **Neo4j**.
    b.  Split, embed, and store the file's content in **Chroma** with the filename as metadata.

This script is the "backend" part. Your LangGraph application (the "frontend") will be a separate script that *queries* these databases, as we discussed.

### Architecture Overview

Here is the data flow for the ingestion pipeline:

`[New .md File] -> Ingestion Script (ingest.py) -> (1) Neo4j (Graph) & (2) Chroma (Vectors)`

-----

### Part 1: Setting Up the Persistent Databases

The easiest way to run persistent databases locally is with Docker. Create a file named `docker-compose.yml` with the following content:

```yaml
version: '3.8'

services:
  # Neo4j Graph Database
  neo4j:
    image: neo4j:5
    container_name: neo4j-db
    ports:
      - "7474:7474"  # Web UI
      - "7687:7687"  # Bolt (Python driver)
    volumes:
      - ./neo4j-data:/data
    environment:
      - NEO4J_AUTH=neo4j/your-strong-password # Change this password
      
  # Chroma Vector Database
  chroma:
    image: chromadb/chroma:latest
    container_name: chroma-db
    ports:
      - "8000:8000"
    volumes:
      - ./chroma-data:/chroma
    command: "chroma run --host 0.0.0.0 --port 8000" # Runs in client-server mode
```

**To run this:**

1.  Save the file as `docker-compose.yml`.
2.  Open your terminal in the same directory.
3.  Run: `docker-compose up -d`

This will start Neo4j (accessible at `http://localhost:7474`) and Chroma (accessible at `http://localhost:8000`). The data will be saved in the `neo4j-data` and `chroma-data` folders, so it persists even if you restart your computer.

-----

### Part 2: The Ingestion Pipeline Code

First, install the necessary Python libraries:

```bash
pip install neo4j chromadb langchain langchain-openai langchain-community langchain-text-splitter
```

Now, create a file named `ingest.py`. This script will contain our entire ingestion logic.

```python
import os
import re
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# --- 1. Configuration & Setup ---

# --- IMPORTANT: Set your OpenAI API Key ---
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

# Database connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your-strong-password"  # Change this to your password

CHROMA_HOST = "localhost"
CHROMA_PORT = "8000"
CHROMA_COLLECTION = "markdown_docs"

# Source directory for your markdown files
DOC_FOLDER = "doc" # This is the same folder from our PoC
TRACKER_FILE = "processed_files.txt" # Simple text file to track ingested files

# Regex to find [[links]]
LINK_PATTERN = re.compile(r"\[\[(.*?)\]\]")

# Setup the embedding function
# This will use your OPENAI_API_KEY
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)

# --- 2. Database Connections ---

def get_neo4j_driver():
    """Establishes connection to the Neo4j database."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Neo4j connection successful.")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def get_chroma_client():
    """Establishes connection to the Chroma database."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat() # Test connection
        print("Chroma connection successful.")
        
        # Get or create the collection. This is where vectors are stored.
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=openai_ef
        )
        return client, collection
    except Exception as e:
        print(f"Failed to connect to Chroma: {e}")
        return None, None

# --- 3. File Processing Logic ---

def process_new_file(filepath, neo4j_driver, chroma_collection):
    """
    Processes a single new file:
    1. Reads content.
    2. Extracts links.
    3. Writes nodes/relationships to Neo4j.
    4. Splits, embeds, and writes chunks to Chroma.
    """
    filename = os.path.basename(filepath)
    print(f"--- Processing: {filename} ---")
    
    try:
        # Load the document content
        loader = TextLoader(filepath)
        doc = loader.load()[0] # Load returns a list, take first item
        content = doc.page_content
        
        # --- 3a. Ingest to Neo4j (Graph) ---
        links = LINK_PATTERN.findall(content)
        
        with neo4j_driver.session() as session:
            # Step 1: Create or update the node for *this* file
            # 'MERGE' is idempotent: it creates if not exist, or matches if it does.
            session.run("MERGE (d:Document {name: $filename})", filename=filename)
            
            # Step 2: Create nodes for all linked documents and the relationship
            for link in links:
                target_name = link if link.endswith(".md") else f"{link}.md"
                
                # Create the target node
                session.run("MERGE (t:Document {name: $target_name})", target_name=target_name)
                
                # Create the relationship
                session.run(
                    """
                    MATCH (d:Document {name: $source_name})
                    MATCH (t:Document {name: $target_name})
                    MERGE (d)-[:LINKS_TO]->(t)
                    """,
                    source_name=filename,
                    target_name=target_name
                )
            print(f"Neo4j: Added node for {filename} with {len(links)} links.")

        # --- 3b. Ingest to Chroma (Vectors) ---
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents([doc])
        
        # Prepare data for Chroma
        documents = []
        metadatas = []
        ids = []
        
        for i, split in enumerate(splits):
            documents.append(split.page_content)
            # This metadata is the "join key" back to our graph
            metadatas.append({"source_filename": filename, "chunk_index": i})
            ids.append(f"{filename}_{i}") # Unique ID for each chunk

        # Add to Chroma collection (this handles embedding automatically)
        if documents:
            chroma_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Chroma: Added {len(documents)} chunks for {filename}.")
            
        return True

    except Exception as e:
        print(f"Failed to process file {filename}: {e}")
        return False

# --- 4. Orchestration (Finding new files) ---

def load_processed_files(tracker_filepath):
    """Loads the set of filenames that have already been processed."""
    if not os.path.exists(tracker_filepath):
        return set()
    with open(tracker_filepath, 'r') as f:
        return set(line.strip() for line in f)

def save_processed_file(tracker_filepath, filename):
    """Adds a new filename to the tracker file."""
    with open(tracker_filepath, 'a') as f:
        f.write(f"{filename}\n")

def run_ingestion():
    """
    Main ingestion loop.
    Scans the DOC_FOLDER, compares against the TRACKER_FILE,
    and processes any new files.
    """
    print("Starting ingestion pipeline...")
    
    # 1. Connect to databases
    neo4j_driver = get_neo4j_driver()
    chroma_client, chroma_collection = get_chroma_client()
    
    if not neo4j_driver or not chroma_client:
        print("Failed to connect to databases. Exiting.")
        return

    # 2. Find new files
    processed_files = load_processed_files(TRACKER_FILE)
    all_files = os.listdir(DOC_FOLDER)
    new_files = [f for f in all_files if f.endswith(".md") and f not in processed_files]
    
    if not new_files:
        print("No new files to process.")
        return

    print(f"Found {len(new_files)} new files to process: {new_files}")
    
    # 3. Process new files
    success_count = 0
    for filename in new_files:
        filepath = os.path.join(DOC_FOLDER, filename)
        if process_new_file(filepath, neo4j_driver, chroma_collection):
            # 4. Update tracker file *after* successful processing
            save_processed_file(TRACKER_FILE, filename)
            success_count += 1
            
    print(f"\nIngestion complete. Processed {success_count} new files.")
    
    # Close connections
    neo4j_driver.close()

# --- 5. Run the Script ---

if __name__ == "__main__":
    # Make sure the 'doc' folder exists
    if not os.path.exists(DOC_FOLDER):
        print(f"Error: '{DOC_FOLDER}' directory not found.")
        print("Please create it and add your 10 mock files from the first example.")
    else:
        run_ingestion()
```

### How to Run This

1.  **Set up your mock files:** Make sure the `doc` folder with your 10 markdown files exists (from our first PoC script).
2.  **Start your databases:** `docker-compose up -d`
3.  **Set your API key:** Add your `OPENAI_API_KEY` at the top of `ingest.py`.
4.  **Run the script:**
    ```bash
    python ingest.py
    ```

**Expected Output (First Run):**

```
Neo4j connection successful.
Chroma connection successful.
Starting ingestion pipeline...
Found 10 new files to process: ['doc1_intro.md', 'doc2_ml.md', ...]
--- Processing: doc1_intro.md ---
Neo4j: Added node for doc1_intro.md with 1 links.
Chroma: Added 1 chunks for doc1_intro.md.
--- Processing: doc2_ml.md ---
Neo4j: Added node for doc2_ml.md with 1 links.
Chroma: Added 1 chunks for doc2_ml.md.
... (processes all 10 files) ...

Ingestion complete. Processed 10 new files.
```

5.  **Run the script again:**

    ```bash
    python ingest.py
    ```

**Expected Output (Second Run):**

```
Neo4j connection successful.
Chroma connection successful.
Starting ingestion pipeline...
No new files to process.
```

Now, if you **add a new file `doc11_new.md`** to the `doc` folder and run `ingest.py` again, it will *only* process that one new file. This is how you handle your "1,000 files a day" requirement.

### Next Steps

Your data is now in persistent, scalable databases. Your `ingest.py` script handles adding new data. The only remaining piece is to **modify your original LangGraph application** to query these databases instead of loading files into memory.

Would you like to see the updated LangGraph `run_graph_rag.py` script that connects to and queries these persistent Neo4j and Chroma databases?
