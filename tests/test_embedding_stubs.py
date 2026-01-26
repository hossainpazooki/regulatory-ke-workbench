"""
Tests for embedding service API stubs (Stories 2-4).

These tests verify that stub endpoints exist and return appropriate NotImplementedError.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestEmbeddingRouteStubs:
    """Test Story 2: Embedding generation route stubs."""

    def test_generate_embeddings_stub_exists(self):
        """Verify generate embeddings endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.embeddings import router

        # Check that the route exists (routes include the prefix)
        routes = [r.path for r in router.routes]
        assert any("embeddings/generate" in r for r in routes)

    def test_get_embeddings_stub_exists(self):
        """Verify get embeddings endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.embeddings import router

        routes = [r.path for r in router.routes]
        assert any("embeddings" in r and "generate" not in r for r in routes)

    def test_delete_embeddings_stub_exists(self):
        """Verify delete embeddings endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.embeddings import router

        # Check by method - delete endpoint exists
        delete_routes = [r for r in router.routes if hasattr(r, 'methods') and 'DELETE' in r.methods]
        assert len(delete_routes) > 0

    def test_embedding_stats_stub_exists(self):
        """Verify stats endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.embeddings import router

        routes = [r.path for r in router.routes]
        assert any("stats" in r for r in routes)


class TestSearchRouteStubs:
    """Test Story 3: Search route stubs."""

    def test_search_by_text_stub_exists(self):
        """Verify text search endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.search import router

        routes = [r.path for r in router.routes]
        assert any("by-text" in r for r in routes)

    def test_search_by_entities_stub_exists(self):
        """Verify entity search endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.search import router

        routes = [r.path for r in router.routes]
        assert any("by-entities" in r for r in routes)

    def test_search_by_outcome_stub_exists(self):
        """Verify outcome search endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.search import router

        routes = [r.path for r in router.routes]
        assert any("by-outcome" in r for r in routes)

    def test_search_by_legal_source_stub_exists(self):
        """Verify legal source search endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.search import router

        routes = [r.path for r in router.routes]
        assert any("by-legal-source" in r for r in routes)

    def test_hybrid_search_stub_exists(self):
        """Verify hybrid search endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.search import router

        routes = [r.path for r in router.routes]
        assert any("hybrid" in r for r in routes)

    def test_similar_rules_stub_exists(self):
        """Verify similar rules endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.search import router

        routes = [r.path for r in router.routes]
        assert any("similar" in r for r in routes)


class TestGraphRouteStubs:
    """Test Story 4: Graph embedding route stubs."""

    def test_get_rule_graph_stub_exists(self):
        """Verify get rule graph endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.graph import router

        routes = [r.path for r in router.routes]
        assert any("graph/rules/{rule_id}" in r and "stats" not in r for r in routes)

    def test_search_by_structure_stub_exists(self):
        """Verify structural search endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.graph import router

        routes = [r.path for r in router.routes]
        assert any("by-structure" in r for r in routes)

    def test_compare_graphs_stub_exists(self):
        """Verify graph comparison endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.graph import router

        routes = [r.path for r in router.routes]
        assert any("compare" in r for r in routes)

    def test_graph_stats_stub_exists(self):
        """Verify graph stats endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.graph import router

        routes = [r.path for r in router.routes]
        assert any("stats" in r for r in routes)

    def test_batch_generate_stub_exists(self):
        """Verify batch generate endpoint is defined."""
        from backend.rule_embedding_service.app.api.routes.graph import router

        routes = [r.path for r in router.routes]
        assert any("batch/generate" in r for r in routes)


class TestGraphEmbeddingService:
    """Test Story 4: Graph embedding service implementation."""

    def test_service_class_exists(self):
        """Verify GraphEmbeddingService class is defined."""
        from backend.embeddings.graph import GraphEmbeddingService

        service = GraphEmbeddingService()
        assert service is not None

    def test_rule_to_graph_implementation(self):
        """Verify rule_to_graph converts rule to NetworkX graph."""
        pytest.importorskip("networkx")
        pytest.importorskip("numpy")
        from backend.embeddings.graph import GraphEmbeddingService
        from backend.embeddings.models import (
            EmbeddingRule,
            EmbeddingCondition,
            EmbeddingDecision,
        )

        service = GraphEmbeddingService()

        # Create a mock rule
        rule = EmbeddingRule(
            id=1,
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            conditions=[
                EmbeddingCondition(
                    id=1,
                    rule_id=1,
                    field="actor.type",
                    operator="equals",
                    value='"retail"',
                )
            ],
            decision=EmbeddingDecision(
                id=1,
                rule_id=1,
                outcome="approved",
                confidence=0.9,
            ),
            legal_sources=[],
        )

        graph = service.rule_to_graph(rule)

        # Verify graph structure
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        # Should have rule node
        assert f"rule:{rule.rule_id}" in graph.nodes()

    def test_generate_graph_embedding_implementation(self):
        """Verify generate_graph_embedding produces embedding vector."""
        nx = pytest.importorskip("networkx")
        pytest.importorskip("numpy")
        from backend.embeddings.graph import GraphEmbeddingService

        service = GraphEmbeddingService()

        # Create simple test graph
        G = nx.Graph()
        G.add_node("rule:test", node_type="rule", label="Test")
        G.add_node("cond:test:0", node_type="condition", label="condition")
        G.add_edge("rule:test", "cond:test:0", edge_type="HAS_CONDITION")

        embedding = service.generate_graph_embedding(G, dimensions=64)

        # Verify embedding shape
        assert len(embedding) == 64
        # Verify non-zero (at least some features)
        assert any(v != 0 for v in embedding)

    def test_find_similar_by_structure_requires_session(self):
        """Verify find_similar_by_structure requires database session."""
        from backend.embeddings.graph import GraphEmbeddingService

        service = GraphEmbeddingService()
        with pytest.raises(ValueError, match="Session required"):
            service.find_similar_by_structure("test_rule")

    def test_compare_graphs_requires_session(self):
        """Verify compare_graphs requires database session."""
        from backend.embeddings.graph import GraphEmbeddingService

        service = GraphEmbeddingService()
        with pytest.raises(ValueError, match="Session required"):
            service.compare_graphs("rule_a", "rule_b")


class TestGraphEmbeddingModel:
    """Test Story 4: GraphEmbedding SQLModel."""

    def test_graph_embedding_model_exists(self):
        """Verify GraphEmbedding model is defined."""
        from backend.embeddings.models import GraphEmbedding

        assert GraphEmbedding is not None
        assert GraphEmbedding.__tablename__ == "graph_embeddings"

    def test_graph_embedding_has_required_fields(self):
        """Verify GraphEmbedding has all required fields."""
        from backend.embeddings.models import GraphEmbedding
        from sqlmodel import Field

        # Check key fields exist
        field_names = list(GraphEmbedding.model_fields.keys())

        assert "id" in field_names
        assert "rule_id" in field_names
        assert "embedding_vector" in field_names
        assert "vector_json" in field_names
        assert "graph_json" in field_names
        assert "num_nodes" in field_names
        assert "num_edges" in field_names
        assert "model_name" in field_names
        assert "walk_length" in field_names
        assert "num_walks" in field_names
        assert "p" in field_names
        assert "q" in field_names

    def test_embedding_rule_has_graph_embeddings_relationship(self):
        """Verify EmbeddingRule has graph_embeddings relationship."""
        from backend.embeddings.models import EmbeddingRule

        # SQLModel relationships are stored in __sqlmodel_relationships__, not model_fields
        assert hasattr(EmbeddingRule, "graph_embeddings")


class TestSearchSchemas:
    """Test Story 3: Search request schemas."""

    def test_text_search_request_schema(self):
        """Verify TextSearchRequest schema."""
        from backend.embeddings.schemas import TextSearchRequest

        request = TextSearchRequest(query="test query", top_k=5)
        assert request.query == "test query"
        assert request.top_k == 5
        assert request.min_similarity == 0.5  # default

    def test_entity_search_request_schema(self):
        """Verify EntitySearchRequest schema."""
        from backend.embeddings.schemas import EntitySearchRequest

        request = EntitySearchRequest(entities=["income", "age"])
        assert request.entities == ["income", "age"]
        assert request.top_k == 10  # default

    def test_outcome_search_request_schema(self):
        """Verify OutcomeSearchRequest schema."""
        from backend.embeddings.schemas import OutcomeSearchRequest

        request = OutcomeSearchRequest(outcome="approved")
        assert request.outcome == "approved"

    def test_legal_source_search_request_schema(self):
        """Verify LegalSourceSearchRequest schema."""
        from backend.embeddings.schemas import LegalSourceSearchRequest

        request = LegalSourceSearchRequest(
            citation="MiCA Article 36",
            document_id="mica_2023",
        )
        assert request.citation == "MiCA Article 36"
        assert request.document_id == "mica_2023"

    def test_hybrid_search_request_schema(self):
        """Verify HybridSearchRequest schema."""
        from backend.embeddings.schemas import HybridSearchRequest

        request = HybridSearchRequest(
            query="income eligibility",
            weights={"semantic": 0.5, "structural": 0.5},
        )
        assert request.query == "income eligibility"
        assert request.weights["semantic"] == 0.5


class TestGraphSchemas:
    """Test Story 4: Graph-related schemas."""

    def test_graph_search_request_schema(self):
        """Verify GraphSearchRequest schema."""
        from backend.embeddings.schemas import GraphSearchRequest

        request = GraphSearchRequest(rule_id="test_rule", top_k=5)
        assert request.rule_id == "test_rule"
        assert request.top_k == 5

    def test_graph_comparison_request_schema(self):
        """Verify GraphComparisonRequest schema."""
        from backend.embeddings.schemas import GraphComparisonRequest

        request = GraphComparisonRequest(rule_id_a="rule_a", rule_id_b="rule_b")
        assert request.rule_id_a == "rule_a"
        assert request.rule_id_b == "rule_b"

    def test_rule_graph_schema(self):
        """Verify RuleGraph schema."""
        from backend.embeddings.schemas import (
            RuleGraph,
            GraphNode,
            GraphEdge,
        )

        graph = RuleGraph(
            rule_id="test",
            nodes=[GraphNode(id="n1", type="rule", label="Test Rule")],
            edges=[GraphEdge(source="n1", target="n2", type="HAS_CONDITION")],
            num_nodes=2,
            num_edges=1,
        )
        assert graph.rule_id == "test"
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 1

    def test_graph_comparison_result_schema(self):
        """Verify GraphComparisonResult schema."""
        from backend.embeddings.schemas import GraphComparisonResult

        result = GraphComparisonResult(
            rule_id_a="a",
            rule_id_b="b",
            similarity_score=0.85,
            common_nodes=5,
            common_edges=4,
            structural_distance=0.15,
        )
        assert result.similarity_score == 0.85
        assert result.common_nodes == 5
