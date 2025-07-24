"""
Utility module to perform retrieval over the local Chroma vector store.

Responsibilities:
- Lazy‑load the embedding model and Chroma collection.
- Embed user queries.
- Execute similarity search to return the most relevant chunks.
"""

from functools import lru_cache
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Reuse the same constants as in load_data.py
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """
    Lazily load and cache the SentenceTransformer model.

    Parameters
    ----------
    None

    Returns
    -------
    SentenceTransformer
        Loaded embedding model instance.

    Functionality
    -------------
    1. Loads the model only once (cached with lru_cache).
    2. Returns the same instance for subsequent calls.
    """
    logger.debug("Loading embedding model %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_collection():
    """
    Lazily load and cache the Chroma collection.

    Parameters
    ----------
    None

    Returns
    -------
    chromadb.api.models.Collection.Collection
        Reference to the Chroma collection used for retrieval.

    Functionality
    -------------
    1. Creates a PersistentClient pointing to the existing ./chroma_db folder.
    2. Gets the collection by name (must have been created by load_data.py).
    3. Caches the handle to avoid repeated disk I/O.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(name=COLLECTION_NAME)


def embed_text(text: str) -> List[float]:
    """
    Compute a dense embedding vector for a single piece of text.

    Parameters
    ----------
    text : str
        Raw input text to embed.

    Returns
    -------
    list[float]
        Embedding vector as a Python list of floats.

    Functionality
    -------------
    1. Retrieves the cached embedding model.
    2. Encodes the input text into a numeric vector.
    3. Converts the NumPy array to a list for JSON serializability.
    """
    model = get_embedder()
    vector = model.encode([text])[0]
    return vector.tolist()


def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Perform a semantic similarity search against the vector store.

    Parameters
    ----------
    query : str
        User's natural language query/question.
    k : int
        Number of top similar chunks to return.

    Returns
    -------
    list[dict]
        List of result objects containing:
        - "text": chunk content
        - "score": distance/similarity value (lower is better for cosine distance)
        - "metadata": original metadata dictionary

    Functionality
    -------------
    1. Embeds the input query.
    2. Runs a similarity search (`query`) over the Chroma collection.
    3. Normalizes the response into a list of dictionaries.
    4. Returns the top-k matches for downstream prompt construction.
    """
    logger.debug("Retrieving top %d chunks for query=%r", k, query)
    collection = get_collection()
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances", "metadatas"]
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Pair up each retrieved chunk with its distance and metadata
    output = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        output.append({
            "text": doc,
            "score": dist,
            "metadata": meta
        })
    return output


def build_context(query: str, k: int = 3) -> str:
    """
    Build a concatenated context string from top retrieved chunks.

    Parameters
    ----------
    query : str
        User question used to drive retrieval.
    k : int
        How many chunks to include.

    Returns
    -------
    str
        A single string containing the retrieved chunks separated by delimiters.

    Functionality
    -------------
    1. Calls `retrieve` to obtain top-k relevant chunks.
    2. Formats each chunk with a simple header delimiter.
    3. Joins them into a context block suitable to inject into the LLM prompt.
    """
    hits = retrieve(query, k=k)
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(f"[Document {i} | score={hit['score']:.4f}]\n{hit['text']}")
    return "\n\n".join(parts)