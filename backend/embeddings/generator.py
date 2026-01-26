"""Embedding generator for rule vectors.

Generates 4 types of embeddings per rule:
- Semantic: from natural language description
- Structural: from serialized conditions/logic
- Entity: from extracted field names
- Legal: from legal source citations

Supports optional ML dependencies (sentence-transformers).
Falls back to simple hash-based vectors when ML is unavailable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import EmbeddingType, EmbeddingRule, RuleEmbedding

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# Default embedding dimension
DEFAULT_DIM = 384

# Try to load sentence-transformers
_encoder: "SentenceTransformer | None" = None
_ml_available = False

try:
    from sentence_transformers import SentenceTransformer
    _ml_available = True
except ImportError:
    pass


def ml_available() -> bool:
    """Check if ML embedding generation is available."""
    return _ml_available


def get_encoder() -> "SentenceTransformer":
    """Get or create the sentence transformer encoder."""
    global _encoder
    if _encoder is None:
        if not _ml_available:
            raise RuntimeError("sentence-transformers not installed")
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


@dataclass
class GeneratedEmbedding:
    """Result of embedding generation."""
    embedding_type: EmbeddingType
    vector: list[float]
    source_text: str
    model_name: str
    vector_dim: int


class EmbeddingGenerator:
    """Generates embeddings for rules.

    Supports two modes:
    - ML mode: Uses sentence-transformers for dense embeddings
    - Fallback mode: Uses hash-based vectors when ML unavailable

    Usage:
        generator = EmbeddingGenerator()
        embeddings = generator.generate_all(rule)

    Direct numpy API (matches spec):
        vector = generator.generate_semantic_embedding("rule description")
        structure = generator.serialize_rule_structure(rule)
        entities = generator.extract_entities(rule)
    """

    def __init__(self, use_ml: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the generator.

        Args:
            use_ml: Whether to use ML embeddings (requires sentence-transformers)
            model_name: Name of the sentence-transformer model
        """
        self.use_ml = use_ml and _ml_available
        self.model_name = model_name if self.use_ml else "hash-fallback"
        self.dim = DEFAULT_DIM

    # =========================================================================
    # Direct numpy API (matches technical requirements spec)
    # =========================================================================

    def generate_semantic_embedding(self, text: str) -> "np.ndarray":
        """Generate semantic embedding from text.

        Args:
            text: Natural language text to embed

        Returns:
            numpy array of shape (384,)
        """
        import numpy as np
        vector = self._encode(text)
        return np.array(vector, dtype=np.float32)

    def generate_structural_embedding(self, rule: EmbeddingRule) -> "np.ndarray":
        """Generate structural embedding from rule logic.

        Args:
            rule: Rule to generate embedding for

        Returns:
            numpy array of shape (384,)
        """
        import numpy as np
        structure = self.serialize_rule_structure(rule)
        vector = self._encode(structure)
        return np.array(vector, dtype=np.float32)

    def serialize_rule_structure(self, rule: EmbeddingRule) -> str:
        """Serialize rule structure to text for embedding.

        Args:
            rule: Rule to serialize

        Returns:
            Serialized string representation of rule logic
        """
        parts = []

        # Serialize conditions
        for cond in rule.conditions:
            parts.append(f"{cond.field} {cond.operator} {cond.value}")

        # Serialize decision
        if rule.decision:
            parts.append(f"OUTCOME:{rule.decision.outcome}")
            parts.append(f"CONFIDENCE:{rule.decision.confidence}")

        return " | ".join(parts) if parts else "EMPTY_STRUCTURE"

    def extract_entities(self, rule: EmbeddingRule) -> list[str]:
        """Extract entity names from rule.

        Args:
            rule: Rule to extract entities from

        Returns:
            List of entity names (fields, operators, outcomes)
        """
        entities = set()

        # Extract field names
        for cond in rule.conditions:
            # Split field paths (e.g., "applicant.income" -> ["applicant", "income"])
            parts = cond.field.replace(".", " ").replace("_", " ").split()
            entities.update(parts)
            entities.add(cond.operator)

        # Add decision outcome
        if rule.decision:
            entities.add(rule.decision.outcome)

        return sorted(entities) if entities else []

    # =========================================================================
    # High-level API (returns GeneratedEmbedding objects)
    # =========================================================================

    def generate_all(self, rule: EmbeddingRule) -> list[GeneratedEmbedding]:
        """Generate all 4 types of embeddings for a rule.

        Args:
            rule: The rule to generate embeddings for

        Returns:
            List of 4 GeneratedEmbedding objects
        """
        return [
            self.generate_semantic(rule),
            self.generate_structural(rule),
            self.generate_entity(rule),
            self.generate_legal(rule),
        ]

    def generate_semantic(self, rule: EmbeddingRule) -> GeneratedEmbedding:
        """Generate semantic embedding from rule description.

        Combines rule name and description into natural language text.
        """
        text_parts = [rule.name]
        if rule.description:
            text_parts.append(rule.description)
        if rule.decision:
            text_parts.append(f"Decision: {rule.decision.outcome}")
            if rule.decision.explanation:
                text_parts.append(rule.decision.explanation)

        source_text = " ".join(text_parts)
        vector = self._encode(source_text)

        return GeneratedEmbedding(
            embedding_type=EmbeddingType.SEMANTIC,
            vector=vector,
            source_text=source_text,
            model_name=self.model_name,
            vector_dim=self.dim,
        )

    def generate_structural(self, rule: EmbeddingRule) -> GeneratedEmbedding:
        """Generate structural embedding from rule logic.

        Serializes conditions and decision structure into text.
        """
        source_text = self.serialize_rule_structure(rule)
        vector = self._encode(source_text)

        return GeneratedEmbedding(
            embedding_type=EmbeddingType.STRUCTURAL,
            vector=vector,
            source_text=source_text,
            model_name=self.model_name,
            vector_dim=self.dim,
        )

    def generate_entity(self, rule: EmbeddingRule) -> GeneratedEmbedding:
        """Generate entity embedding from field names.

        Extracts field names and operators for entity-based matching.
        """
        entities = self.extract_entities(rule)
        source_text = " ".join(entities) if entities else "EMPTY_ENTITIES"
        vector = self._encode(source_text)

        return GeneratedEmbedding(
            embedding_type=EmbeddingType.ENTITY,
            vector=vector,
            source_text=source_text,
            model_name=self.model_name,
            vector_dim=self.dim,
        )

    def generate_legal(self, rule: EmbeddingRule) -> GeneratedEmbedding:
        """Generate legal embedding from citations.

        Combines legal source citations and document references.
        """
        parts = []

        for source in rule.legal_sources:
            parts.append(source.citation)
            if source.document_id:
                parts.append(source.document_id)

        source_text = " ".join(parts) if parts else "NO_LEGAL_SOURCES"
        vector = self._encode(source_text)

        return GeneratedEmbedding(
            embedding_type=EmbeddingType.LEGAL,
            vector=vector,
            source_text=source_text,
            model_name=self.model_name,
            vector_dim=self.dim,
        )

    def _encode(self, text: str) -> list[float]:
        """Encode text to a vector.

        Uses ML encoder if available, otherwise falls back to hash-based vector.
        """
        if self.use_ml:
            return self._encode_ml(text)
        return self._encode_hash(text)

    def _encode_ml(self, text: str) -> list[float]:
        """Encode using sentence-transformers."""
        encoder = get_encoder()
        embedding = encoder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _encode_hash(self, text: str) -> list[float]:
        """Fallback: Generate hash-based pseudo-embedding.

        Creates a deterministic vector from text hash.
        Not suitable for semantic similarity, but maintains API compatibility.
        """
        # Create a deterministic vector from hash
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Extend hash to fill vector dimension
        vector = []
        seed = int.from_bytes(hash_bytes[:8], "big")
        for i in range(self.dim):
            # Simple pseudo-random based on hash
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            # Normalize to [-1, 1]
            value = (seed / 0x7FFFFFFF) * 2 - 1
            vector.append(value)

        return vector


def create_embedding_records(
    rule_db_id: int,
    embeddings: list[GeneratedEmbedding],
) -> list[RuleEmbedding]:
    """Create RuleEmbedding database records from generated embeddings.

    Args:
        rule_db_id: Database ID of the rule
        embeddings: List of generated embeddings

    Returns:
        List of RuleEmbedding objects ready for database insertion
    """
    records = []
    for emb in embeddings:
        record = RuleEmbedding(
            rule_id=rule_db_id,
            embedding_type=emb.embedding_type.value,
            vector_json=json.dumps(emb.vector),
            vector_dim=emb.vector_dim,
            model_name=emb.model_name,
            source_text=emb.source_text[:1000] if emb.source_text else None,  # Truncate long text
        )
        records.append(record)
    return records
