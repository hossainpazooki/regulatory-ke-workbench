"""
Analytics API client for Streamlit frontend.

Provides a high-level interface for the analytics API endpoints,
supporting rule comparison, clustering, similarity search, and visualization.
"""

from __future__ import annotations

from typing import Any
import requests

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"


class AnalyticsClient:
    """Client for analytics API endpoints."""

    def __init__(self, base_url: str = DEFAULT_API_URL):
        """Initialize the analytics client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a GET request."""
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: dict) -> dict:
        """Make a POST request."""
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Summary & Health
    # =========================================================================

    def get_summary(self) -> dict:
        """Get analytics summary statistics.

        Returns:
            Dict with total_rules, jurisdictions, embedding_types_available, etc.
        """
        return self._get("/analytics/summary")

    # =========================================================================
    # Rule Comparison
    # =========================================================================

    def compare_rules(
        self,
        rule1_id: str,
        rule2_id: str,
        weights: dict[str, float] | None = None,
    ) -> dict:
        """Compare two rules across all embedding types.

        Args:
            rule1_id: First rule ID
            rule2_id: Second rule ID
            weights: Optional per-type weights (e.g., {"semantic": 0.5, "entity": 0.3})

        Returns:
            ComparisonResult dict with overall_similarity, similarity_by_type, etc.
        """
        data = {
            "rule1_id": rule1_id,
            "rule2_id": rule2_id,
        }
        if weights:
            data["weights"] = weights
        return self._post("/analytics/rules/compare", data)

    # =========================================================================
    # Clustering
    # =========================================================================

    def get_clusters(
        self,
        embedding_type: str = "semantic",
        n_clusters: int | None = None,
        algorithm: str = "kmeans",
    ) -> dict:
        """Get rule clusters.

        Args:
            embedding_type: Type of embedding (semantic, structural, entity, legal)
            n_clusters: Number of clusters (auto-detect if None)
            algorithm: Clustering algorithm (kmeans, dbscan, hierarchical)

        Returns:
            ClusterAnalysis dict with num_clusters, clusters, silhouette_score, etc.
        """
        params = {
            "embedding_type": embedding_type,
            "algorithm": algorithm,
        }
        if n_clusters is not None:
            params["n_clusters"] = n_clusters
        return self._get("/analytics/rule-clusters", params)

    def cluster_rules(
        self,
        embedding_type: str = "semantic",
        n_clusters: int | None = None,
        algorithm: str = "kmeans",
        rule_ids: list[str] | None = None,
    ) -> dict:
        """Cluster rules with full options (POST variant).

        Args:
            embedding_type: Type of embedding
            n_clusters: Number of clusters
            algorithm: Clustering algorithm
            rule_ids: Optional subset of rules to cluster

        Returns:
            ClusterAnalysis dict
        """
        data = {
            "embedding_type": embedding_type,
            "algorithm": algorithm,
        }
        if n_clusters is not None:
            data["n_clusters"] = n_clusters
        if rule_ids:
            data["rule_ids"] = rule_ids
        return self._post("/analytics/rule-clusters", data)

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    def find_conflicts(
        self,
        rule_ids: list[str] | None = None,
        conflict_types: list[str] | None = None,
        threshold: float = 0.7,
    ) -> dict:
        """Find potential rule conflicts.

        Args:
            rule_ids: Specific rules to analyze (all if None)
            conflict_types: Types of conflicts to detect (semantic, structural, jurisdiction)
            threshold: Similarity threshold for conflict detection

        Returns:
            ConflictReport dict with total_rules_analyzed, conflicts_found, conflicts list
        """
        data = {
            "threshold": threshold,
        }
        if rule_ids:
            data["rule_ids"] = rule_ids
        if conflict_types:
            data["conflict_types"] = conflict_types
        return self._post("/analytics/find-conflicts", data)

    def get_conflicts(self, threshold: float = 0.7) -> dict:
        """Get all detected conflicts (simplified GET variant).

        Args:
            threshold: Similarity threshold

        Returns:
            ConflictReport dict
        """
        return self._get("/analytics/conflicts", {"threshold": threshold})

    # =========================================================================
    # Similarity Search
    # =========================================================================

    def get_similar_rules(
        self,
        rule_id: str,
        embedding_type: str = "all",
        top_k: int = 10,
        min_score: float = 0.5,
        include_explanation: bool = True,
    ) -> dict:
        """Find rules similar to a given rule.

        Args:
            rule_id: Query rule ID
            embedding_type: Type of embedding (all, semantic, structural, entity, legal)
            top_k: Maximum number of results
            min_score: Minimum similarity score
            include_explanation: Whether to include explanations

        Returns:
            SimilarRulesResponse dict with query_rule_id, similar_rules list
        """
        params = {
            "embedding_type": embedding_type,
            "top_k": top_k,
            "min_score": min_score,
            "include_explanation": include_explanation,
        }
        return self._get(f"/analytics/rules/{rule_id}/similar", params)

    # =========================================================================
    # Coverage Analysis
    # =========================================================================

    def get_coverage(self) -> dict:
        """Get legal source coverage analysis.

        Returns:
            CoverageReport dict with total_rules, coverage_by_framework, coverage_gaps
        """
        return self._get("/analytics/coverage")

    # =========================================================================
    # UMAP Projection
    # =========================================================================

    def get_umap_projection(
        self,
        embedding_type: str = "semantic",
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> dict:
        """Get UMAP projection of rule embeddings.

        Args:
            embedding_type: Type of embedding to project
            n_components: Number of dimensions (2 or 3)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter

        Returns:
            UMAPProjectionResponse dict with points list, each containing x, y, (z)
        """
        params = {
            "embedding_type": embedding_type,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        }
        return self._get("/analytics/umap-projection", params)

    def get_umap_projection_with_options(
        self,
        embedding_type: str = "semantic",
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        rule_ids: list[str] | None = None,
    ) -> dict:
        """Get UMAP projection with full options (POST variant).

        Args:
            embedding_type: Type of embedding
            n_components: Dimensions
            n_neighbors: UMAP parameter
            min_dist: UMAP parameter
            rule_ids: Subset of rules to project

        Returns:
            UMAPProjectionResponse dict
        """
        data = {
            "embedding_type": embedding_type,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        }
        if rule_ids:
            data["rule_ids"] = rule_ids
        return self._post("/analytics/umap-projection", data)


# Global client instance
_client: AnalyticsClient | None = None


def get_analytics_client(base_url: str = DEFAULT_API_URL) -> AnalyticsClient:
    """Get or create the global analytics client.

    Args:
        base_url: API base URL

    Returns:
        AnalyticsClient instance
    """
    global _client
    if _client is None:
        _client = AnalyticsClient(base_url)
    return _client


def reset_analytics_client() -> None:
    """Reset the global analytics client."""
    global _client
    _client = None
