"""Analytics schemas for rule comparison, clustering, and conflict detection."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class EmbeddingTypeEnum(str, Enum):
    """Embedding types for analytics."""

    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    ENTITY = "entity"
    LEGAL = "legal"
    GRAPH = "graph"
    ALL = "all"


class ClusterAlgorithm(str, Enum):
    """Clustering algorithms."""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"


class ConflictType(str, Enum):
    """Types of rule conflicts."""

    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    JURISDICTION = "jurisdiction"


class ConflictSeverity(str, Enum):
    """Conflict severity levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CoverageImportance(str, Enum):
    """Coverage gap importance levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Rule Comparison Schemas
# ============================================================================


class CompareRulesRequest(BaseModel):
    """Request to compare two rules."""

    rule1_id: str = Field(..., description="First rule ID")
    rule2_id: str = Field(..., description="Second rule ID")
    include_graph: bool = Field(True, description="Include graph structure comparison")
    weights: dict[str, float] | None = Field(
        None,
        description="Optional weights per embedding type (semantic, structural, entity, legal)",
    )


class ComparisonResult(BaseModel):
    """Result of comparing two rules."""

    rule1_id: str
    rule2_id: str
    rule1_name: str | None = None
    rule2_name: str | None = None
    overall_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted average similarity"
    )
    similarity_by_type: dict[str, float] = Field(
        ..., description="Similarity score per embedding type"
    )
    structural_comparison: dict[str, Any] = Field(
        default_factory=dict, description="Graph structure metrics"
    )
    shared_entities: list[str] = Field(
        default_factory=list, description="Common field names"
    )
    shared_legal_sources: list[str] = Field(
        default_factory=list, description="Common citations"
    )
    conflict_indicators: list[str] = Field(
        default_factory=list, description="Potential conflicts detected"
    )


# ============================================================================
# Clustering Schemas
# ============================================================================


class ClusterRequest(BaseModel):
    """Request for rule clustering."""

    embedding_type: EmbeddingTypeEnum = Field(
        EmbeddingTypeEnum.SEMANTIC, description="Embedding type to cluster on"
    )
    n_clusters: int | None = Field(
        None, ge=2, le=50, description="Number of clusters (auto-detect if None)"
    )
    algorithm: ClusterAlgorithm = Field(
        ClusterAlgorithm.KMEANS, description="Clustering algorithm"
    )
    rule_ids: list[str] | None = Field(
        None, description="Specific rules to cluster (all if None)"
    )


class ClusterInfo(BaseModel):
    """Information about a single cluster."""

    cluster_id: int
    size: int = Field(..., ge=0, description="Number of rules in cluster")
    rule_ids: list[str] = Field(..., description="Rules in this cluster")
    centroid_rule_id: str | None = Field(
        None, description="Most representative rule (closest to centroid)"
    )
    cohesion_score: float = Field(
        ..., ge=0.0, le=1.0, description="Intra-cluster similarity"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Common terms in cluster"
    )


class ClusterAnalysis(BaseModel):
    """Result of clustering analysis."""

    num_clusters: int = Field(..., ge=0, description="Number of clusters found")
    algorithm: ClusterAlgorithm
    embedding_type: EmbeddingTypeEnum
    silhouette_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Quality metric (-1 to 1)"
    )
    clusters: list[ClusterInfo] = Field(..., description="Cluster details")
    total_rules: int = Field(..., ge=0, description="Total rules analyzed")


# ============================================================================
# Conflict Detection Schemas
# ============================================================================


class ConflictSearchRequest(BaseModel):
    """Request to find rule conflicts."""

    rule_ids: list[str] | None = Field(
        None, description="Specific rules to check (all if None)"
    )
    conflict_types: list[ConflictType] = Field(
        default=[ConflictType.SEMANTIC, ConflictType.STRUCTURAL],
        description="Types of conflicts to detect",
    )
    threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Similarity threshold for conflict detection"
    )


class ConflictInfo(BaseModel):
    """Information about a detected conflict."""

    rule1_id: str
    rule2_id: str
    rule1_name: str | None = None
    rule2_name: str | None = None
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str = Field(..., description="Human-readable description")
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    conflicting_aspects: list[str] = Field(
        default_factory=list, description="What specifically conflicts"
    )
    resolution_hints: list[str] = Field(
        default_factory=list, description="Suggestions for resolution"
    )


class ConflictReport(BaseModel):
    """Report of detected conflicts."""

    total_rules_analyzed: int = Field(..., ge=0)
    conflicts_found: int = Field(..., ge=0)
    conflicts: list[ConflictInfo] = Field(default_factory=list)
    high_severity_count: int = Field(0, ge=0)
    medium_severity_count: int = Field(0, ge=0)
    low_severity_count: int = Field(0, ge=0)


# ============================================================================
# Similarity Search Schemas
# ============================================================================


class SimilarityExplanation(BaseModel):
    """Explanation of why two rules are similar."""

    primary_reason: str = Field(..., description="Main similarity reason")
    shared_entities: list[str] = Field(
        default_factory=list, description="Common field names"
    )
    shared_legal_sources: list[str] = Field(
        default_factory=list, description="Common citations"
    )
    structural_similarity: str | None = Field(
        None, description="Decision tree structure comparison"
    )
    semantic_alignment: str | None = Field(
        None, description="Semantic content alignment"
    )


class SimilarRule(BaseModel):
    """A rule similar to the query rule."""

    rule_id: str
    rule_name: str | None = None
    jurisdiction: str | None = None
    overall_score: float = Field(..., ge=0.0, le=1.0)
    scores_by_type: dict[str, float] = Field(
        default_factory=dict, description="Similarity per embedding type"
    )
    explanation: SimilarityExplanation | None = None


class SimilarRulesRequest(BaseModel):
    """Request to find similar rules."""

    rule_id: str = Field(..., description="Query rule ID")
    embedding_type: EmbeddingTypeEnum = Field(
        EmbeddingTypeEnum.ALL, description="Embedding type to search"
    )
    top_k: int = Field(10, ge=1, le=100, description="Maximum results")
    min_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity")
    include_explanation: bool = Field(True, description="Include explanations")


class SimilarRulesResponse(BaseModel):
    """Response with similar rules."""

    query_rule_id: str
    query_rule_name: str | None = None
    similar_rules: list[SimilarRule] = Field(default_factory=list)
    total_candidates: int = Field(0, ge=0, description="Total rules searched")


# ============================================================================
# Coverage Analysis Schemas
# ============================================================================


class FrameworkCoverage(BaseModel):
    """Coverage statistics for a legal framework."""

    framework: str = Field(..., description="Framework name (MiCA, FCA, etc.)")
    total_articles: int = Field(..., ge=0)
    covered_articles: int = Field(..., ge=0)
    coverage_percentage: float = Field(..., ge=0.0, le=100.0)
    rules_per_article: dict[str, int] = Field(
        default_factory=dict, description="Article -> rule count"
    )
    rule_count: int = Field(0, ge=0, description="Total rules for this framework")


class CoverageGap(BaseModel):
    """A gap in legal source coverage."""

    framework: str
    article: str
    importance: CoverageImportance
    recommendation: str


class CoverageReport(BaseModel):
    """Report on legal source coverage."""

    total_rules: int = Field(..., ge=0)
    total_legal_sources: int = Field(..., ge=0)
    coverage_by_framework: dict[str, FrameworkCoverage] = Field(default_factory=dict)
    uncovered_sources: list[str] = Field(
        default_factory=list, description="Legal sources without rules"
    )
    coverage_gaps: list[CoverageGap] = Field(
        default_factory=list, description="Identified coverage gaps"
    )
    overall_coverage_percentage: float = Field(0.0, ge=0.0, le=100.0)


# ============================================================================
# UMAP Projection Schemas (for visualization)
# ============================================================================


class UMAPPoint(BaseModel):
    """A single point in UMAP space."""

    rule_id: str
    rule_name: str | None = None
    x: float
    y: float
    z: float | None = None  # For 3D
    jurisdiction: str | None = None
    cluster_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UMAPProjectionRequest(BaseModel):
    """Request for UMAP projection."""

    embedding_type: EmbeddingTypeEnum = Field(EmbeddingTypeEnum.SEMANTIC)
    n_components: int = Field(2, ge=2, le=3, description="2D or 3D")
    n_neighbors: int = Field(15, ge=2, le=100)
    min_dist: float = Field(0.1, ge=0.0, le=1.0)
    rule_ids: list[str] | None = Field(None, description="Specific rules (all if None)")


class UMAPProjectionResponse(BaseModel):
    """Response with UMAP projection."""

    points: list[UMAPPoint] = Field(default_factory=list)
    embedding_type: EmbeddingTypeEnum
    n_components: int
    total_rules: int = Field(0, ge=0)
