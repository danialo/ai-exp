"""Vector store wrapper for semantic search.

Provides abstraction over ChromaDB for storing and querying embeddings.
Supports upsert operations and similarity search with metadata filtering.
"""

from pathlib import Path
from typing import Any, Optional

import chromadb
import numpy as np
from chromadb.config import Settings


class VectorStoreResult:
    """Result from vector store query."""

    def __init__(self, id: str, score: float, metadata: Optional[dict[str, Any]] = None):
        """Initialize result.

        Args:
            id: Document/experience ID
            score: Similarity score (higher = more similar)
            metadata: Associated metadata
        """
        self.id = id
        self.score = score
        self.metadata = metadata or {}


class VectorStore:
    """Vector store for semantic embeddings.

    Wraps ChromaDB for efficient similarity search and metadata filtering.
    Supports multiple collections for different embedding types (prompt, response, etc.).
    """

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "experiences",
        reset: bool = False,
    ):
        """Initialize vector store.

        Args:
            persist_directory: Directory to persist vector index
            collection_name: Name of the collection
            reset: If True, delete existing collection and start fresh
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
            except ValueError:
                pass  # Collection doesn't exist

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def upsert(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Insert or update a vector.

        Args:
            id: Unique identifier for the vector
            vector: Embedding vector
            metadata: Optional metadata to store with vector
        """
        # Convert numpy array to list for ChromaDB
        vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

        # Ensure metadata values are JSON-serializable
        clean_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)

        self.collection.upsert(
            ids=[id],
            embeddings=[vector_list],
            metadatas=[clean_metadata] if clean_metadata else None,
        )

    def upsert_batch(
        self,
        ids: list[str],
        vectors: list[np.ndarray],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Batch insert or update vectors.

        Args:
            ids: List of unique identifiers
            vectors: List of embedding vectors
            metadatas: Optional list of metadata dicts
        """
        # Convert numpy arrays to lists
        vector_lists = [v.tolist() if isinstance(v, np.ndarray) else v for v in vectors]

        # Clean metadata
        clean_metadatas = None
        if metadatas:
            clean_metadatas = []
            for meta in metadatas:
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                clean_metadatas.append(clean_meta)

        self.collection.upsert(
            ids=ids,
            embeddings=vector_lists,
            metadatas=clean_metadatas,
        )

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[VectorStoreResult]:
        """Query for similar vectors.

        Args:
            vector: Query vector
            top_k: Number of results to return
            where: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of VectorStoreResults, ordered by similarity (highest first)
        """
        vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

        results = self.collection.query(
            query_embeddings=[vector_list],
            n_results=top_k,
            where=where,
        )

        # Parse results
        output = []
        if results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

            for id_, dist, meta in zip(ids, distances, metadatas):
                # Convert distance to similarity score (1 - cosine distance)
                score = 1.0 - dist
                output.append(VectorStoreResult(id=id_, score=score, metadata=meta))

        return output

    def get(self, id: str) -> Optional[VectorStoreResult]:
        """Get a specific vector by ID.

        Args:
            id: Vector ID

        Returns:
            VectorStoreResult if found, None otherwise
        """
        try:
            results = self.collection.get(ids=[id], include=["metadatas"])

            if results["ids"] and len(results["ids"]) > 0:
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                return VectorStoreResult(id=id, score=1.0, metadata=metadata)
        except Exception:
            pass

        return None

    def delete(self, id: str) -> None:
        """Delete a vector by ID.

        Args:
            id: Vector ID to delete
        """
        self.collection.delete(ids=[id])

    def count(self) -> int:
        """Count total vectors in collection.

        Returns:
            Number of vectors
        """
        return self.collection.count()

    def reset(self) -> None:
        """Delete all vectors in collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def create_vector_store(
    persist_directory: Optional[str | Path] = None,
    collection_name: str = "experiences",
    reset: bool = False,
) -> VectorStore:
    """Factory function to create vector store.

    Args:
        persist_directory: Directory for persistence (defaults to data/vector_index/)
        collection_name: Collection name
        reset: Reset collection if it exists

    Returns:
        VectorStore instance
    """
    if persist_directory is None:
        persist_directory = Path("data/vector_index/")

    return VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        reset=reset,
    )
