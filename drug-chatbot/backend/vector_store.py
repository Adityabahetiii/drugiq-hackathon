"""
Vector Store (ChromaDB)
-----------------------
Uses HuggingFace sentence-transformers for embeddings (free, runs locally)
and ChromaDB to store and search chunks.

Enhanced with:
  - Incremental indexing (add one PDF at a time)
  - File tracking (prevent duplicate indexing)
  - Drug removal (delete a drug's data from the store)
  - Multi-drug search capabilities
"""

import os
import ssl

# Fix SSL certificate verification issues on some Windows systems
# This prevents "[SSL: CERTIFICATE_VERIFY_FAILED]" errors when downloading
# the sentence-transformer model from HuggingFace

# 1. Fix Python's default SSL context
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 2. Tell HuggingFace Hub to skip SSL verification (it uses its own HTTP client)
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''

# 3. Tell requests library to skip SSL verification
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''

import chromadb
from chromadb.utils import embedding_functions
from backend.pdf_processor import load_pdfs, load_single_pdf


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "drug_info"


def get_embedding_function():
    """
    Uses HuggingFace's all-MiniLM-L6-v2 model — free, runs on your laptop,
    no API key needed. Downloads once (~90MB), then works offline.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def build_vector_store(pdf_folder: str):
    """
    Reads all PDFs, chunks them, embeds them, and stores in ChromaDB.
    Call this once to set up the database.
    """
    print("Loading PDFs and creating chunks...")
    chunks = load_pdfs(pdf_folder)

    if not chunks:
        raise ValueError("No chunks found. Make sure PDFs are in the data/pdfs/ folder.")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Cleared existing collection.")
    except Exception:
        pass

    embedding_fn = get_embedding_function()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding and storing {len(chunks)} chunks...")

    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "drug_name": c["drug_name"],
            "page_number": c["page_number"],
            "source_file": c["source_file"]
        }
        for c in chunks
    ]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"  Stored {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")

    print("Vector store built successfully!")
    return collection


def get_vector_store():
    """Loads an already-built ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    return collection


def add_pdf_to_store(pdf_path: str, collection=None):
    """
    Indexes a single new PDF into the existing ChromaDB collection.
    This allows incremental updates without rebuilding the entire database.

    Returns the updated collection.
    """
    if collection is None:
        collection = get_vector_store()

    filename = os.path.basename(pdf_path)

    # Check if this file is already indexed
    indexed_files = get_indexed_files(collection)
    if filename in indexed_files:
        print(f"{filename} is already indexed. Removing old data first...")
        remove_file_from_store(filename, collection)

    # Process the single PDF
    chunks = load_single_pdf(pdf_path)

    if not chunks:
        raise ValueError(f"No chunks extracted from {filename}. The PDF might be empty or unreadable.")

    # Generate unique IDs based on existing count
    existing_count = collection.count()
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "drug_name": c["drug_name"],
            "page_number": c["page_number"],
            "source_file": c["source_file"]
        }
        for c in chunks
    ]
    ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]

    # Add in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    drug_name = chunks[0]["drug_name"] if chunks else filename
    print(f"Added {len(chunks)} chunks from {filename} ({drug_name}) to the vector store.")
    return collection


def get_indexed_files(collection) -> list[str]:
    """Returns a list of unique source filenames already indexed in the collection."""
    try:
        all_data = collection.get(include=["metadatas"])
        files = list({m["source_file"] for m in all_data["metadatas"]})
        return sorted(files)
    except Exception:
        return []


def get_drug_info(collection) -> list[dict]:
    """
    Returns detailed info about all indexed drugs:
    [{drug_name, source_file, chunk_count, pages}]
    """
    try:
        all_data = collection.get(include=["metadatas"])
        drug_map = {}
        for m in all_data["metadatas"]:
            key = m["source_file"]
            if key not in drug_map:
                drug_map[key] = {
                    "drug_name": m["drug_name"],
                    "source_file": m["source_file"],
                    "chunk_count": 0,
                    "pages": set()
                }
            drug_map[key]["chunk_count"] += 1
            drug_map[key]["pages"].add(m["page_number"])

        result = []
        for info in drug_map.values():
            result.append({
                "drug_name": info["drug_name"],
                "source_file": info["source_file"],
                "chunk_count": info["chunk_count"],
                "page_count": len(info["pages"])
            })
        return sorted(result, key=lambda x: x["drug_name"])
    except Exception:
        return []


def remove_file_from_store(filename: str, collection):
    """
    Removes all chunks associated with a specific source file from the collection.
    """
    all_data = collection.get(include=["metadatas"])
    ids_to_remove = []
    for i, m in enumerate(all_data["metadatas"]):
        if m["source_file"] == filename:
            ids_to_remove.append(all_data["ids"][i])

    if ids_to_remove:
        # ChromaDB delete in batches
        batch_size = 100
        for i in range(0, len(ids_to_remove), batch_size):
            collection.delete(ids=ids_to_remove[i:i+batch_size])
        print(f"Removed {len(ids_to_remove)} chunks for {filename}")


def remove_drug_from_store(drug_name: str, collection):
    """
    Removes all chunks associated with a specific drug name from the collection.
    """
    all_data = collection.get(include=["metadatas"])
    ids_to_remove = []
    for i, m in enumerate(all_data["metadatas"]):
        if m["drug_name"].lower() == drug_name.lower():
            ids_to_remove.append(all_data["ids"][i])

    if ids_to_remove:
        batch_size = 100
        for i in range(0, len(ids_to_remove), batch_size):
            collection.delete(ids=ids_to_remove[i:i+batch_size])
        print(f"Removed {len(ids_to_remove)} chunks for drug: {drug_name}")


def search(collection, query: str, top_k: int = 4, where: dict | None = None) -> list[dict]:
    """
    Searches the vector store for chunks most relevant to the query.
    Pass `where` (e.g. {"drug_name": "Rinvoq"}) to scope retrieval to a
    single drug's chunks once the query has been matched to one — this
    keeps citations from bleeding into unrelated drugs' pages.
    """
    query_kwargs = {"query_texts": [query], "n_results": top_k}
    if where:
        query_kwargs["where"] = where
    results = collection.query(**query_kwargs)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "drug_name": results["metadatas"][0][i]["drug_name"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "source_file": results["metadatas"][0][i]["source_file"],
            "distance": results["distances"][0][i] if results.get("distances") else None
        })

    return chunks


def search_multi_drug(collection, query: str, top_k: int = 8) -> list[dict]:
    """
    Enhanced search that retrieves chunks across multiple drugs.
    Useful for condition-based queries like "What treats rheumatoid arthritis?"
    Returns more chunks to cover multiple drugs.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "drug_name": results["metadatas"][0][i]["drug_name"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "source_file": results["metadatas"][0][i]["source_file"],
            "distance": results["distances"][0][i] if results.get("distances") else None
        })

    return chunks


def get_all_drug_names(collection) -> list[str]:
    """Returns all unique drug names in the database."""
    all_data = collection.get(include=["metadatas"])
    drug_names = list({m["drug_name"] for m in all_data["metadatas"]})
    return sorted(drug_names)


def vector_store_exists() -> bool:
    """Check if ChromaDB has already been built."""
    return os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0
