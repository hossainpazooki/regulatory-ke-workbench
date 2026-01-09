"""BM25 retrieval implementation (no ML dependencies)."""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class BM25Document:
    """A document in the BM25 index."""

    id: str
    text: str
    tokens: list[str]
    metadata: dict


class BM25Index:
    """BM25 index for text retrieval."""

    def __init__(self):
        self._documents: list[BM25Document] = []
        self._index: BM25Okapi | None = None

    def add_documents(self, documents: list[dict]) -> None:
        """Add documents to the index.

        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'.
        """
        for doc in documents:
            tokens = self._tokenize(doc["text"])
            self._documents.append(
                BM25Document(
                    id=doc["id"],
                    text=doc["text"],
                    tokens=tokens,
                    metadata=doc.get("metadata", {}),
                )
            )

        # Rebuild index
        if self._documents:
            corpus = [doc.tokens for doc in self._documents]
            self._index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> list[tuple[BM25Document, float]]:
        """Search the index.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of (document, score) tuples.
        """
        if not self._index or not self._documents:
            return []

        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)

        # Get top-k results
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = indexed_scores[:top_k]

        results = []
        for idx, score in top_results:
            if score > 0:
                results.append((self._documents[idx], score))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25."""
        # Lowercase
        text = text.lower()
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Split and filter empty
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens

    def __len__(self) -> int:
        return len(self._documents)
