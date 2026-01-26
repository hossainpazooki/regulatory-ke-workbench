"""SQLModel table definitions for embedding rules.

Supports 4 types of embeddings per rule:
- Semantic: from natural language description
- Structural: from serialized conditions/logic
- Entity: from extracted field names
- Legal: from legal source citations
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


class EmbeddingType(str, Enum):
    """Types of embeddings generated per rule."""
    SEMANTIC = "semantic"       # From natural language description
    STRUCTURAL = "structural"   # From serialized conditions/logic
    ENTITY = "entity"           # From extracted field names
    LEGAL = "legal"             # From legal source citations


class RuleEmbedding(SQLModel, table=True):
    """Vector embeddings for a rule.

    Stores 4 types of embeddings per rule for multi-faceted similarity search:
    - semantic: Dense embedding from rule description/name
    - structural: Embedding from serialized decision logic
    - entity: Embedding from field names and operators
    - legal: Embedding from legal citations and sources

    Storage options:
    - SQLite: vector_json (JSON string)
    - PostgreSQL: embedding_vector (bytes) or pgvector extension
    """

    __tablename__ = "rule_embeddings"

    id: Optional[int] = Field(default=None, primary_key=True)
    rule_id: int = Field(foreign_key="embedding_rules.id", index=True, ondelete="CASCADE")
    embedding_type: str = Field(..., index=True, description="Type of embedding (semantic/structural/entity/legal)")

    # Primary storage: JSON string (SQLite compatible)
    vector_json: str = Field(..., description="Embedding vector as JSON array")

    # Alternative storage: bytes (serialized numpy array)
    # For production PostgreSQL, use: embedding_vector = Column(LargeBinary)
    embedding_vector: Optional[bytes] = Field(default=None, description="Serialized numpy array (optional)")

    vector_dim: int = Field(default=384, description="Dimension of the vector")

    # Metadata
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Model used to generate embedding")
    source_text: Optional[str] = Field(default=None, description="Text used to generate embedding")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )

    rule: Optional["EmbeddingRule"] = Relationship(back_populates="embeddings")

    def get_vector_as_numpy(self) -> "np.ndarray":
        """Get embedding vector as numpy array.

        Returns:
            numpy array of shape (vector_dim,)
        """
        import json
        import numpy as np

        # Prefer bytes if available
        if self.embedding_vector:
            return np.frombuffer(self.embedding_vector, dtype=np.float32)

        # Fall back to JSON
        return np.array(json.loads(self.vector_json), dtype=np.float32)

    def set_vector_from_numpy(self, arr: "np.ndarray") -> None:
        """Set embedding vector from numpy array.

        Args:
            arr: numpy array to store
        """
        import json
        import numpy as np

        # Store in both formats for compatibility
        self.vector_json = json.dumps(arr.tolist())
        self.embedding_vector = arr.astype(np.float32).tobytes()
        self.vector_dim = len(arr)


class GraphEmbedding(SQLModel, table=True):
    """Graph-based embedding for structural similarity using Node2Vec.

    Stores graph structure and Node2Vec embeddings for rules,
    enabling structural similarity search independent of semantics.
    """

    __tablename__ = "graph_embeddings"

    id: Optional[int] = Field(default=None, primary_key=True)
    rule_id: int = Field(foreign_key="embedding_rules.id", index=True, ondelete="CASCADE")

    # Vector storage (dual format like RuleEmbedding)
    embedding_vector: bytes = Field(..., description="Serialized numpy array")
    vector_json: Optional[str] = Field(default=None, description="JSON array fallback")
    vector_dim: int = Field(default=128, description="Node2Vec typically uses 128 dims")

    # Graph structure
    graph_json: str = Field(..., description="NetworkX node-link JSON format")
    num_nodes: int = Field(..., description="Number of nodes in graph")
    num_edges: int = Field(..., description="Number of edges in graph")

    # Model metadata (for reproducibility)
    model_name: str = Field(default="node2vec")
    walk_length: int = Field(default=80, description="Random walk length")
    num_walks: int = Field(default=10, description="Walks per node")
    p: float = Field(default=1.0, description="Return parameter")
    q: float = Field(default=1.0, description="In-out parameter")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )

    rule: Optional["EmbeddingRule"] = Relationship(back_populates="graph_embeddings")

    def get_vector_as_numpy(self) -> "np.ndarray":
        """Get embedding vector as numpy array."""
        import json
        import numpy as np

        if self.embedding_vector:
            return np.frombuffer(self.embedding_vector, dtype=np.float32)
        if self.vector_json:
            return np.array(json.loads(self.vector_json), dtype=np.float32)
        raise ValueError("No embedding vector available")

    def set_vector_from_numpy(self, arr: "np.ndarray") -> None:
        """Set embedding vector from numpy array."""
        import json
        import numpy as np

        self.vector_json = json.dumps(arr.tolist())
        self.embedding_vector = arr.astype(np.float32).tobytes()
        self.vector_dim = len(arr)


class EmbeddingRule(SQLModel, table=True):
    """Rule entity for embedding-based search."""

    __tablename__ = "embedding_rules"

    id: Optional[int] = Field(default=None, primary_key=True)
    rule_id: str = Field(..., unique=True, index=True, description="Human-readable rule ID")
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(default=None, description="Rule description")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )
    is_active: bool = Field(default=True, description="Whether the rule is active")

    conditions: list["EmbeddingCondition"] = Relationship(
        back_populates="rule",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"},
    )
    decision: Optional["EmbeddingDecision"] = Relationship(
        back_populates="rule",
        sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan", "lazy": "selectin"},
    )
    legal_sources: list["EmbeddingLegalSource"] = Relationship(
        back_populates="rule",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"},
    )
    embeddings: list["RuleEmbedding"] = Relationship(
        back_populates="rule",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"},
    )
    graph_embeddings: list["GraphEmbedding"] = Relationship(
        back_populates="rule",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"},
    )


class EmbeddingCondition(SQLModel, table=True):
    """Rule condition for embedding rules."""

    __tablename__ = "embedding_conditions"

    id: Optional[int] = Field(default=None, primary_key=True)
    field: str = Field(..., description="Field path to check")
    operator: str = Field(..., description="Comparison operator")
    value: str = Field(..., description="JSON-serialized comparison value")
    description: Optional[str] = Field(default=None, description="Human-readable description")

    rule_id: int = Field(foreign_key="embedding_rules.id", ondelete="CASCADE")
    rule: Optional["EmbeddingRule"] = Relationship(back_populates="conditions")


class EmbeddingDecision(SQLModel, table=True):
    """Rule decision for embedding rules."""

    __tablename__ = "embedding_decisions"

    id: Optional[int] = Field(default=None, primary_key=True)
    outcome: str = Field(..., description="Decision outcome")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    explanation: Optional[str] = Field(default=None, description="Explanation")

    rule_id: int = Field(foreign_key="embedding_rules.id", unique=True, ondelete="CASCADE")
    rule: Optional["EmbeddingRule"] = Relationship(back_populates="decision")


class EmbeddingLegalSource(SQLModel, table=True):
    """Legal source for embedding rules."""

    __tablename__ = "embedding_legal_sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    citation: str = Field(..., description="Legal citation")
    document_id: Optional[str] = Field(default=None, description="Document identifier")
    url: Optional[str] = Field(default=None, description="URL to source")

    rule_id: int = Field(foreign_key="embedding_rules.id", ondelete="CASCADE")
    rule: Optional["EmbeddingRule"] = Relationship(back_populates="legal_sources")


RuleEmbedding.model_rebuild()
GraphEmbedding.model_rebuild()
EmbeddingRule.model_rebuild()
EmbeddingCondition.model_rebuild()
EmbeddingDecision.model_rebuild()
EmbeddingLegalSource.model_rebuild()
