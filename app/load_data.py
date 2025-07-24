"""
Build a local vector store for Retrieval-Augmented Generation (RAG).

Steps:
1. Read the raw knowledge base file.
2. Split the text into overlapping chunks.
3. Generate dense embeddings with a SentenceTransformer model.
4. Persist the chunks + embeddings into a local ChromaDB collection.

Run:
    python -m app.load_data
"""

from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from app.logging_config import configure_logging

# ---------------------------
# Configuration constants
# ---------------------------
CHROMA_PATH = "chroma_db"                          # Folder where Chroma will persist data
COLLECTION_NAME = "knowledge"                      # Name of the Chroma collection
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embedding model
SOURCE_FILE = Path("knowledge_base.txt")           # Raw knowledge base file (must exist)

configure_logging()
logger = logging.getLogger(__name__)

def read_source_text() -> str:
    """
    Read the full contents of the knowledge base file.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Full text from the knowledge base file.

    Functionality
    -------------
    1. Verifies the file exists.
    2. Reads the file using UTF-8 encoding.
    3. Raises FileNotFoundError if missing.
    """
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base file not found: {SOURCE_FILE.resolve()}"
        )
    return SOURCE_FILE.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Split the input text into overlapping chunks.

    Parameters
    ----------
    text : str
        Original full input text.
    chunk_size : int
        Target size in characters for each chunk.
    overlap : int
        Number of characters to repeat from the previous chunk for context.

    Returns
    -------
    list[str]
        List of non-empty chunk strings.

    Functionality
    -------------
    1. Iteratively extracts windows of size `chunk_size`.
    2. Moves the start index by `chunk_size - overlap` to preserve context.
    3. Strips whitespace and filters out empty chunks.
    4. Returns the list ready for embedding.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # advance with overlap
    return [c for c in chunks if c]


def main():
    """
    Orchestrate the embedding + storage process.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Functionality
    -------------
    1. Load the SentenceTransformer embedding model.
    2. Read and chunk the knowledge base text.
    3. Initialize (or recreate) a persistent Chroma collection.
    4. Compute embeddings for each chunk.
    5. Store documents, embeddings, and metadata in Chroma.
    6. Print progress messages for transparency.
    """
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer(MODEL_NAME)

    logger.info("Reading knowledge base file...")
    full_text = read_source_text()

    logger.info("Splitting text into chunks...")
    chunks = chunk_text(full_text)
    if len(chunks) < 2:
        logger.warning("Only one chunk was produced. Consider adding more text.")
    logger.info("Generated %d chunks.", len(chunks))

    logger.info("Initializing Chroma persistent client...")
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Previous collection '%s' deleted.", COLLECTION_NAME)
    except Exception:
        logger.debug("No previous collection to delete.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    logger.info("Computing embeddings...")
    embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=True)

    ids = [f"doc_{i}" for i in range(len(chunks))]
    metadatas = [{"source": f"chunk_{i}"} for i in range(len(chunks))]

    logger.info("Adding documents to Chroma collection...")
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=metadatas
    )

    logger.info("Success! Vector store created at ./chroma_db")
    logger.info("You can now run the API server and perform retrieval.")


if __name__ == "__main__":
    main()