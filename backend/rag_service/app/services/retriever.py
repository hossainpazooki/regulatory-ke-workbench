"""Hybrid retriever with BM25 and optional vector search."""

from dataclasses import dataclass
from pathlib import Path

from backend.config import ml_available, get_settings
from .bm25 import BM25Index, BM25Document
from .chunker import Chunk


@dataclass
class RetrievalResult:
    """A retrieval result with source information."""

    chunk_id: str
    text: str
    score: float
    document_id: str
    metadata: dict
    retrieval_method: str  # "bm25", "vector", or "hybrid"


class Retriever:
    """Hybrid retriever combining BM25 and optional vector search."""

    def __init__(self, use_vectors: bool | None = None):
        """Initialize the retriever.

        Args:
            use_vectors: Whether to use vector search. If None, auto-detect.
        """
        self._bm25 = BM25Index()
        self._vector_store = None
        self._embedder = None
        self._collection = None

        # Auto-detect ML availability
        if use_vectors is None:
            settings = get_settings()
            use_vectors = settings.enable_vector_search and ml_available()

        if use_vectors:
            self._init_vector_store()

    def _init_vector_store(self) -> None:
        """Initialize vector store if ML dependencies available."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._vector_store = chromadb.Client()
            self._collection = self._vector_store.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            pass

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index.

        Args:
            chunks: List of Chunk objects.
        """
        # Add to BM25
        bm25_docs = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            }
            for chunk in chunks
        ]
        self._bm25.add_documents(bm25_docs)

        # Add to vector store if available
        if self._collection is not None and self._embedder is not None:
            texts = [chunk.text for chunk in chunks]
            embeddings = self._embedder.encode(texts).tolist()

            self._collection.add(
                ids=[chunk.id for chunk in chunks],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "document_id": chunk.document_id,
                        "chunk_index": str(chunk.chunk_index),
                    }
                    for chunk in chunks
                ],
            )

    def add_documents(self, documents: list[dict]) -> None:
        """Add raw documents (will be chunked internally).

        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'.
        """
        from .chunker import chunk_text

        all_chunks = []
        for doc in documents:
            chunks = chunk_text(
                text=doc["text"],
                document_id=doc["id"],
                metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)

        self.add_chunks(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> list[RetrievalResult]:
        """Search for relevant chunks.

        Args:
            query: Search query.
            top_k: Number of results to return.
            method: "bm25", "vector", or "hybrid".

        Returns:
            List of RetrievalResult objects.
        """
        results = []

        if method in ("bm25", "hybrid"):
            bm25_results = self._search_bm25(query, top_k)
            results.extend(bm25_results)

        if method in ("vector", "hybrid") and self._collection is not None:
            vector_results = self._search_vector(query, top_k)
            results.extend(vector_results)

        # Deduplicate and sort by score
        seen_ids = set()
        unique_results = []
        for result in sorted(results, key=lambda x: x.score, reverse=True):
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                unique_results.append(result)

        return unique_results[:top_k]

    def _search_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Search using BM25."""
        results = []
        for doc, score in self._bm25.search(query, top_k):
            results.append(
                RetrievalResult(
                    chunk_id=doc.id,
                    text=doc.text,
                    score=score,
                    document_id=doc.metadata.get("document_id", ""),
                    metadata=doc.metadata,
                    retrieval_method="bm25",
                )
            )
        return results

    def _search_vector(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Search using vector similarity."""
        if self._collection is None or self._embedder is None:
            return []

        query_embedding = self._embedder.encode([query]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        retrieval_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity

                retrieval_results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        text=results["documents"][0][i] if results["documents"] else "",
                        score=score,
                        document_id=results["metadatas"][0][i].get("document_id", "")
                        if results["metadatas"]
                        else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        retrieval_method="vector",
                    )
                )

        return retrieval_results

    def __len__(self) -> int:
        return len(self._bm25)
