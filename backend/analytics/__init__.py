"""Analytics domain - rule comparison, clustering, similarity search."""

from .router import router
from .service import RuleAnalyticsService
from .schemas import (
    # Enums
    EmbeddingTypeEnum,
    ClusterAlgorithm,
    ConflictType,
    ConflictSeverity,
    CoverageImportance,
    # Rule Comparison
    CompareRulesRequest,
    ComparisonResult,
    # Clustering
    ClusterRequest,
    ClusterInfo,
    ClusterAnalysis,
    # Conflict Detection
    ConflictSearchRequest,
    ConflictInfo,
    ConflictReport,
    # Similarity Search
    SimilarityExplanation,
    SimilarRule,
    SimilarRulesRequest,
    SimilarRulesResponse,
    # Coverage Analysis
    FrameworkCoverage,
    CoverageGap,
    CoverageReport,
    # UMAP Projection
    UMAPPoint,
    UMAPProjectionRequest,
    UMAPProjectionResponse,
)

# Drift detection
from .drift import (
    DriftDetector,
    DriftReport,
    DriftMetrics,
)

# Error pattern analysis
from .error_patterns import (
    ErrorPatternAnalyzer,
    ErrorPattern,
    CategoryStats,
    ReviewQueueItem as AnalyticsReviewQueueItem,
)

__all__ = [
    # Router
    "router",
    # Service
    "RuleAnalyticsService",
    # Enums
    "EmbeddingTypeEnum",
    "ClusterAlgorithm",
    "ConflictType",
    "ConflictSeverity",
    "CoverageImportance",
    # Rule Comparison
    "CompareRulesRequest",
    "ComparisonResult",
    # Clustering
    "ClusterRequest",
    "ClusterInfo",
    "ClusterAnalysis",
    # Conflict Detection
    "ConflictSearchRequest",
    "ConflictInfo",
    "ConflictReport",
    # Similarity Search
    "SimilarityExplanation",
    "SimilarRule",
    "SimilarRulesRequest",
    "SimilarRulesResponse",
    # Coverage Analysis
    "FrameworkCoverage",
    "CoverageGap",
    "CoverageReport",
    # UMAP Projection
    "UMAPPoint",
    "UMAPProjectionRequest",
    "UMAPProjectionResponse",
    # Error Patterns
    "ErrorPatternAnalyzer",
    "ErrorPattern",
    "CategoryStats",
    "AnalyticsReviewQueueItem",
    # Drift Detection
    "DriftDetector",
    "DriftReport",
    "DriftMetrics",
]
