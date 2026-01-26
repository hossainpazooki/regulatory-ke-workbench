"""Schemas for embedding and graph stores."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field
import uuid


class EmbeddingType(str, Enum):
    """Type of embedding vector."""

    SEMANTIC = "semantic"  # Meaning-based
    STRUCTURAL = "structural"  # Tree/logic structure
    ENTITY = "entity"  # Named entities
    LEGAL = "legal"  # Legal concept based
    GRAPH = "graph"  # Node2Vec graph embedding


class EmbeddingRecord(BaseModel):
    """A stored embedding vector with metadata."""

    id: str = Field(default_factory=lambda: f"emb_{uuid.uuid4().hex[:12]}")
    rule_id: str
    embedding_type: EmbeddingType
    vector: list[float]
    dimension: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Metadata
    model_name: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""

        use_enum_values = True


class GraphNode(BaseModel):
    """A node in the rule graph."""

    id: str = Field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    node_type: Literal["rule", "condition", "obligation", "outcome", "entity", "concept"]
    label: str
    rule_id: str | None = None

    # Node properties
    properties: dict[str, Any] = Field(default_factory=dict)

    # Embedding (if computed)
    embedding: list[float] | None = None
    embedding_model: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphEdge(BaseModel):
    """An edge connecting two nodes in the rule graph."""

    id: str = Field(default_factory=lambda: f"edge_{uuid.uuid4().hex[:8]}")
    source_id: str
    target_id: str
    edge_type: Literal[
        "has_condition",
        "leads_to",
        "requires",
        "references",
        "conflicts_with",
        "supersedes",
        "related_to",
    ]
    weight: float = 1.0

    # Edge properties
    properties: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphQuery(BaseModel):
    """Query for graph traversal."""

    # Start node(s)
    start_node_ids: list[str] | None = None
    start_node_type: str | None = None
    start_rule_id: str | None = None

    # Traversal parameters
    edge_types: list[str] | None = None
    max_depth: int = 2
    direction: Literal["outgoing", "incoming", "both"] = "both"

    # Filtering
    node_types: list[str] | None = None
    min_weight: float | None = None

    # Limiting
    limit: int = 100


class GraphQueryResult(BaseModel):
    """Result of a graph query."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    paths: list[list[str]] | None = None  # List of node ID paths

    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    query_time_ms: int = 0


class SimilaritySearchRequest(BaseModel):
    """Request for similarity search."""

    query_vector: list[float] | None = None
    query_text: str | None = None  # Will be embedded
    rule_id: str | None = None  # Use existing rule's embedding
    embedding_type: EmbeddingType = EmbeddingType.SEMANTIC
    top_k: int = 10
    threshold: float = 0.0  # Minimum similarity


class SimilaritySearchResult(BaseModel):
    """Result of similarity search."""

    rule_id: str
    similarity: float
    embedding_type: EmbeddingType
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphStats(BaseModel):
    """Statistics about the graph store."""

    total_nodes: int = 0
    total_edges: int = 0
    nodes_by_type: dict[str, int] = Field(default_factory=dict)
    edges_by_type: dict[str, int] = Field(default_factory=dict)
    rules_with_embeddings: int = 0
    avg_edges_per_node: float = 0.0
