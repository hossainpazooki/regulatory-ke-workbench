"""Embedding Store - manages rule embedding vectors."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import (
    EmbeddingRecord,
    EmbeddingType,
    SimilaritySearchRequest,
    SimilaritySearchResult,
)

if TYPE_CHECKING:
    pass


class EmbeddingStore:
    """In-memory store for rule embeddings with persistence.

    Supports:
    - CRUD operations for embeddings
    - Similarity search (cosine similarity)
    - Multiple embedding types per rule
    - JSON file persistence
    """

    def __init__(self, persist_path: str | Path | None = None):
        """Initialize the embedding store.

        Args:
            persist_path: Optional path for JSON persistence
        """
        self._embeddings: dict[str, EmbeddingRecord] = {}
        self._by_rule: dict[str, dict[EmbeddingType, str]] = {}  # rule_id -> type -> emb_id
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    def add(self, record: EmbeddingRecord) -> str:
        """Add an embedding record.

        Args:
            record: The embedding to store

        Returns:
            The embedding ID
        """
        # Validate dimension matches vector length
        if len(record.vector) != record.dimension:
            record.dimension = len(record.vector)

        self._embeddings[record.id] = record

        # Index by rule and type
        if record.rule_id not in self._by_rule:
            self._by_rule[record.rule_id] = {}
        self._by_rule[record.rule_id][record.embedding_type] = record.id

        self._persist()
        return record.id

    def get(self, embedding_id: str) -> EmbeddingRecord | None:
        """Get an embedding by ID.

        Args:
            embedding_id: The embedding ID

        Returns:
            The embedding record or None
        """
        return self._embeddings.get(embedding_id)

    def get_by_rule(
        self,
        rule_id: str,
        embedding_type: EmbeddingType | None = None,
    ) -> list[EmbeddingRecord]:
        """Get embeddings for a rule.

        Args:
            rule_id: The rule ID
            embedding_type: Optional filter by type

        Returns:
            List of embedding records
        """
        if rule_id not in self._by_rule:
            return []

        type_map = self._by_rule[rule_id]

        if embedding_type:
            emb_id = type_map.get(embedding_type)
            if emb_id and emb_id in self._embeddings:
                return [self._embeddings[emb_id]]
            return []

        # Return all types
        return [
            self._embeddings[emb_id]
            for emb_id in type_map.values()
            if emb_id in self._embeddings
        ]

    def update(self, record: EmbeddingRecord) -> bool:
        """Update an existing embedding.

        Args:
            record: The updated embedding record

        Returns:
            True if updated, False if not found
        """
        if record.id not in self._embeddings:
            return False

        record.updated_at = datetime.utcnow()
        self._embeddings[record.id] = record

        # Update index
        if record.rule_id not in self._by_rule:
            self._by_rule[record.rule_id] = {}
        self._by_rule[record.rule_id][record.embedding_type] = record.id

        self._persist()
        return True

    def delete(self, embedding_id: str) -> bool:
        """Delete an embedding.

        Args:
            embedding_id: The embedding ID

        Returns:
            True if deleted, False if not found
        """
        if embedding_id not in self._embeddings:
            return False

        record = self._embeddings.pop(embedding_id)

        # Remove from index
        if record.rule_id in self._by_rule:
            type_map = self._by_rule[record.rule_id]
            if record.embedding_type in type_map:
                del type_map[record.embedding_type]
            if not type_map:
                del self._by_rule[record.rule_id]

        self._persist()
        return True

    def delete_by_rule(self, rule_id: str) -> int:
        """Delete all embeddings for a rule.

        Args:
            rule_id: The rule ID

        Returns:
            Number of embeddings deleted
        """
        if rule_id not in self._by_rule:
            return 0

        count = 0
        for emb_id in list(self._by_rule[rule_id].values()):
            if emb_id in self._embeddings:
                del self._embeddings[emb_id]
                count += 1

        del self._by_rule[rule_id]
        self._persist()
        return count

    def search(
        self,
        request: SimilaritySearchRequest,
    ) -> list[SimilaritySearchResult]:
        """Search for similar embeddings.

        Args:
            request: The search request

        Returns:
            List of similar embeddings with scores
        """
        # Get query vector
        query_vector = request.query_vector

        if query_vector is None and request.rule_id:
            # Use existing rule's embedding
            records = self.get_by_rule(request.rule_id, request.embedding_type)
            if records:
                query_vector = records[0].vector

        if query_vector is None:
            return []

        # Search all embeddings of the requested type
        results: list[tuple[float, EmbeddingRecord]] = []

        for record in self._embeddings.values():
            if record.embedding_type != request.embedding_type:
                continue

            # Skip the query rule itself
            if request.rule_id and record.rule_id == request.rule_id:
                continue

            # Compute cosine similarity
            similarity = self._cosine_similarity(query_vector, record.vector)

            if similarity >= request.threshold:
                results.append((similarity, record))

        # Sort by similarity descending
        results.sort(key=lambda x: x[0], reverse=True)

        # Return top-k
        return [
            SimilaritySearchResult(
                rule_id=record.rule_id,
                similarity=sim,
                embedding_type=record.embedding_type,
                metadata=record.metadata,
            )
            for sim, record in results[: request.top_k]
        ]

    def search_by_vector(
        self,
        vector: list[float],
        embedding_type: EmbeddingType = EmbeddingType.SEMANTIC,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[SimilaritySearchResult]:
        """Convenience method for vector search.

        Args:
            vector: The query vector
            embedding_type: Type of embeddings to search
            top_k: Number of results
            threshold: Minimum similarity

        Returns:
            List of similar embeddings
        """
        return self.search(
            SimilaritySearchRequest(
                query_vector=vector,
                embedding_type=embedding_type,
                top_k=top_k,
                threshold=threshold,
            )
        )

    def list_rules(self) -> list[str]:
        """List all rules with embeddings.

        Returns:
            List of rule IDs
        """
        return list(self._by_rule.keys())

    def count(self) -> int:
        """Get total embedding count.

        Returns:
            Number of embeddings
        """
        return len(self._embeddings)

    def count_by_type(self) -> dict[str, int]:
        """Get embedding counts by type.

        Returns:
            Dict of type -> count
        """
        counts: dict[str, int] = {}
        for record in self._embeddings.values():
            type_name = record.embedding_type.value if isinstance(record.embedding_type, EmbeddingType) else record.embedding_type
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (0 to 1)
        """
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _persist(self) -> None:
        """Persist embeddings to disk."""
        if not self._persist_path:
            return

        data = {
            "embeddings": [
                record.model_dump(mode="json")
                for record in self._embeddings.values()
            ]
        }

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        """Load embeddings from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return

        try:
            with open(self._persist_path) as f:
                data = json.load(f)

            for emb_data in data.get("embeddings", []):
                record = EmbeddingRecord(**emb_data)
                self._embeddings[record.id] = record

                if record.rule_id not in self._by_rule:
                    self._by_rule[record.rule_id] = {}
                self._by_rule[record.rule_id][record.embedding_type] = record.id

        except Exception:
            pass  # Ignore load errors, start fresh
