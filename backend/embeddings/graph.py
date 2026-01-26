"""Graph-based rule embedding service.

Uses NetworkX and Node2Vec for structural similarity analysis.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlmodel import Session

from .models import (
    EmbeddingRule,
    GraphEmbedding,
)


class GraphEmbeddingService:
    """Graph-based rule embedding using NetworkX and Node2Vec.

    Converts rules to graph structures and generates embeddings
    that capture structural relationships between rule components.

    Graph Structure:
    - Nodes: rule, conditions, entities, decisions, legal sources
    - Edges: HAS_CONDITION, REFERENCES_ENTITY, PRODUCES_DECISION, CITES_SOURCE
    """

    def __init__(self, session: "Session | None" = None):
        """Initialize the graph embedding service.

        Args:
            session: SQLModel session for database operations
        """
        self.session = session
        self._nx = None
        self._np = None

    @property
    def nx(self):
        """Lazy-load networkx."""
        if self._nx is None:
            try:
                import networkx as nx
                self._nx = nx
            except ImportError:
                raise ImportError(
                    "networkx is required for graph embeddings. "
                    "Install with: pip install networkx"
                )
        return self._nx

    @property
    def np(self):
        """Lazy-load numpy."""
        if self._np is None:
            try:
                import numpy as np
                self._np = np
            except ImportError:
                raise ImportError(
                    "numpy is required for graph embeddings. "
                    "Install with: pip install numpy"
                )
        return self._np

    def rule_to_graph(self, rule: EmbeddingRule) -> Any:
        """Convert a rule to a NetworkX graph.

        Creates a graph representation with:
        - Central rule node
        - Condition nodes connected to rule
        - Entity nodes extracted from conditions
        - Decision node connected to rule
        - Legal source nodes connected to rule

        Args:
            rule: The EmbeddingRule to convert

        Returns:
            NetworkX Graph representing the rule structure
        """
        nx = self.nx
        G = nx.Graph()

        # Add central rule node
        rule_node_id = f"rule:{rule.rule_id}"
        G.add_node(
            rule_node_id,
            node_type="rule",
            label=rule.name,
            rule_id=rule.rule_id,
        )

        # Add condition nodes
        for i, condition in enumerate(rule.conditions):
            cond_node_id = f"cond:{rule.rule_id}:{i}"
            G.add_node(
                cond_node_id,
                node_type="condition",
                label=f"{condition.field} {condition.operator}",
                field=condition.field,
                operator=condition.operator,
            )
            G.add_edge(
                rule_node_id,
                cond_node_id,
                edge_type="HAS_CONDITION",
                weight=1.0,
            )

            # Extract entity nodes from field paths
            field_parts = condition.field.split(".")
            for j, part in enumerate(field_parts):
                entity_node_id = f"entity:{part}"
                if not G.has_node(entity_node_id):
                    G.add_node(
                        entity_node_id,
                        node_type="entity",
                        label=part,
                    )
                G.add_edge(
                    cond_node_id,
                    entity_node_id,
                    edge_type="REFERENCES_ENTITY",
                    weight=0.5,
                )

            # Extract value entities if JSON
            try:
                value = json.loads(condition.value)
                if isinstance(value, str):
                    value_node_id = f"value:{value}"
                    if not G.has_node(value_node_id):
                        G.add_node(
                            value_node_id,
                            node_type="value",
                            label=str(value)[:30],
                        )
                    G.add_edge(
                        cond_node_id,
                        value_node_id,
                        edge_type="HAS_VALUE",
                        weight=0.3,
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        # Add decision node
        if rule.decision:
            decision_node_id = f"decision:{rule.rule_id}"
            G.add_node(
                decision_node_id,
                node_type="decision",
                label=rule.decision.outcome,
                confidence=rule.decision.confidence,
            )
            G.add_edge(
                rule_node_id,
                decision_node_id,
                edge_type="PRODUCES_DECISION",
                weight=1.0,
            )

        # Add legal source nodes
        for i, source in enumerate(rule.legal_sources):
            source_node_id = f"source:{rule.rule_id}:{i}"
            G.add_node(
                source_node_id,
                node_type="legal_source",
                label=source.citation[:50] if source.citation else "Unknown",
                document_id=source.document_id,
            )
            G.add_edge(
                rule_node_id,
                source_node_id,
                edge_type="CITES_SOURCE",
                weight=0.8,
            )

        return G

    def generate_graph_embedding(
        self,
        graph: Any,
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
    ) -> Any:
        """Generate an embedding for a graph using Node2Vec.

        Node2Vec performs biased random walks on the graph and uses
        Skip-gram to learn node embeddings. The graph embedding is
        computed as the mean of all node embeddings.

        Args:
            graph: NetworkX graph to embed
            dimensions: Embedding dimensionality
            walk_length: Length of random walks
            num_walks: Number of walks per node
            p: Return parameter (controls likelihood of revisiting nodes)
            q: In-out parameter (controls search behavior: BFS vs DFS)

        Returns:
            numpy array of shape (dimensions,) representing the graph
        """
        np = self.np

        if graph.number_of_nodes() == 0:
            return np.zeros(dimensions, dtype=np.float32)

        # Try to use node2vec if available
        try:
            from node2vec import Node2Vec

            # Generate walks and learn embeddings
            node2vec = Node2Vec(
                graph,
                dimensions=dimensions,
                walk_length=walk_length,
                num_walks=num_walks,
                p=p,
                q=q,
                workers=1,
                quiet=True,
            )
            model = node2vec.fit(window=10, min_count=1, batch_words=4)

            # Aggregate node embeddings (mean pooling)
            embeddings = []
            for node in graph.nodes():
                try:
                    embeddings.append(model.wv[str(node)])
                except KeyError:
                    continue

            if embeddings:
                return np.mean(embeddings, axis=0).astype(np.float32)
            return np.zeros(dimensions, dtype=np.float32)

        except ImportError:
            # Fallback: use simple structural features
            return self._fallback_embedding(graph, dimensions)

    def _fallback_embedding(self, graph: Any, dimensions: int) -> Any:
        """Generate embedding using structural features when Node2Vec unavailable.

        Uses graph metrics to create a deterministic embedding.
        """
        np = self.np
        nx = self.nx

        features = []

        # Basic graph metrics
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        features.extend([n_nodes, n_edges])

        # Node type counts
        node_types = ["rule", "condition", "entity", "value", "decision", "legal_source"]
        for ntype in node_types:
            count = sum(
                1 for _, data in graph.nodes(data=True)
                if data.get("node_type") == ntype
            )
            features.append(count)

        # Edge type counts
        edge_types = [
            "HAS_CONDITION", "REFERENCES_ENTITY", "HAS_VALUE",
            "PRODUCES_DECISION", "CITES_SOURCE"
        ]
        for etype in edge_types:
            count = sum(
                1 for _, _, data in graph.edges(data=True)
                if data.get("edge_type") == etype
            )
            features.append(count)

        # Degree statistics
        if n_nodes > 0:
            degrees = [d for _, d in graph.degree()]
            features.extend([
                np.mean(degrees),
                np.std(degrees) if len(degrees) > 1 else 0,
                max(degrees),
                min(degrees),
            ])
        else:
            features.extend([0, 0, 0, 0])

        # Density and clustering
        features.append(nx.density(graph))
        try:
            features.append(nx.average_clustering(graph))
        except Exception:
            features.append(0)

        # Pad or truncate to target dimensions
        features = np.array(features, dtype=np.float32)

        if len(features) < dimensions:
            # Pad with zeros
            padded = np.zeros(dimensions, dtype=np.float32)
            padded[:len(features)] = features
            return padded
        else:
            # Hash-based dimensionality reduction
            result = np.zeros(dimensions, dtype=np.float32)
            for i, f in enumerate(features):
                result[i % dimensions] += f
            # Normalize
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
            return result

    def find_similar_by_structure(
        self,
        query_rule_id: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find rules with similar graph structures.

        Computes structural similarity between the query rule's graph
        and all other rules in the database.

        Args:
            query_rule_id: Reference rule to find similar rules for
            top_k: Number of results to return

        Returns:
            List of dicts with rule_id, similarity_score, etc.

        Raises:
            ValueError: If query rule is not found
        """
        from sqlmodel import select

        if not self.session:
            raise ValueError("Session required for database operations")

        np = self.np

        # Get query rule's graph embedding
        stmt = select(GraphEmbedding).join(EmbeddingRule).where(
            EmbeddingRule.rule_id == query_rule_id
        )
        query_embedding = self.session.exec(stmt).first()

        if not query_embedding:
            raise ValueError(f"No graph embedding found for rule: {query_rule_id}")

        query_vector = query_embedding.get_vector_as_numpy()

        # Get all other embeddings
        stmt = select(GraphEmbedding, EmbeddingRule).join(EmbeddingRule).where(
            EmbeddingRule.rule_id != query_rule_id
        )
        results = self.session.exec(stmt).all()

        similarities = []
        for emb, rule in results:
            try:
                other_vector = emb.get_vector_as_numpy()
                similarity = self._cosine_similarity(query_vector, other_vector)
                similarities.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "similarity_score": float(similarity),
                    "num_nodes": emb.num_nodes,
                    "num_edges": emb.num_edges,
                })
            except Exception:
                continue

        # Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:top_k]

    def compare_graphs(
        self,
        rule_id_a: str,
        rule_id_b: str,
    ) -> dict[str, Any]:
        """Compare the graph structures of two rules.

        Analyzes:
        - Graph edit distance
        - Common nodes and edges
        - Structural patterns
        - Embedding similarity

        Args:
            rule_id_a: First rule ID
            rule_id_b: Second rule ID

        Returns:
            Comparison result with similarity metrics

        Raises:
            ValueError: If either rule is not found
        """
        from sqlmodel import select

        if not self.session:
            raise ValueError("Session required for database operations")

        np = self.np
        nx = self.nx

        # Get both rules
        stmt = select(EmbeddingRule).where(EmbeddingRule.rule_id == rule_id_a)
        rule_a = self.session.exec(stmt).first()
        if not rule_a:
            raise ValueError(f"Rule not found: {rule_id_a}")

        stmt = select(EmbeddingRule).where(EmbeddingRule.rule_id == rule_id_b)
        rule_b = self.session.exec(stmt).first()
        if not rule_b:
            raise ValueError(f"Rule not found: {rule_id_b}")

        # Convert to graphs
        graph_a = self.rule_to_graph(rule_a)
        graph_b = self.rule_to_graph(rule_b)

        # Compute metrics
        result = {
            "rule_a": {
                "rule_id": rule_id_a,
                "num_nodes": graph_a.number_of_nodes(),
                "num_edges": graph_a.number_of_edges(),
            },
            "rule_b": {
                "rule_id": rule_id_b,
                "num_nodes": graph_b.number_of_nodes(),
                "num_edges": graph_b.number_of_edges(),
            },
        }

        # Node type comparison
        def get_node_types(g):
            types = {}
            for _, data in g.nodes(data=True):
                ntype = data.get("node_type", "unknown")
                types[ntype] = types.get(ntype, 0) + 1
            return types

        types_a = get_node_types(graph_a)
        types_b = get_node_types(graph_b)
        all_types = set(types_a.keys()) | set(types_b.keys())

        result["node_type_comparison"] = {
            ntype: {
                "rule_a": types_a.get(ntype, 0),
                "rule_b": types_b.get(ntype, 0),
            }
            for ntype in all_types
        }

        # Common entities
        entities_a = {
            data.get("label")
            for _, data in graph_a.nodes(data=True)
            if data.get("node_type") == "entity"
        }
        entities_b = {
            data.get("label")
            for _, data in graph_b.nodes(data=True)
            if data.get("node_type") == "entity"
        }

        common_entities = entities_a & entities_b
        result["common_entities"] = list(common_entities)
        result["entity_jaccard"] = (
            len(common_entities) / len(entities_a | entities_b)
            if entities_a | entities_b else 0
        )

        # Embedding similarity
        emb_a = self.generate_graph_embedding(graph_a)
        emb_b = self.generate_graph_embedding(graph_b)
        result["embedding_similarity"] = float(self._cosine_similarity(emb_a, emb_b))

        # Structural similarity (based on metrics)
        metrics_a = [
            graph_a.number_of_nodes(),
            graph_a.number_of_edges(),
            nx.density(graph_a),
        ]
        metrics_b = [
            graph_b.number_of_nodes(),
            graph_b.number_of_edges(),
            nx.density(graph_b),
        ]
        metrics_a = np.array(metrics_a, dtype=np.float32)
        metrics_b = np.array(metrics_b, dtype=np.float32)
        result["structural_similarity"] = float(self._cosine_similarity(metrics_a, metrics_b))

        return result

    def get_rule_graph_stats(self, rule_id: str) -> dict[str, Any]:
        """Get statistics about a rule's graph structure.

        Args:
            rule_id: The rule identifier

        Returns:
            Dict with num_nodes, num_edges, node_types, etc.

        Raises:
            ValueError: If rule is not found
        """
        from sqlmodel import select

        if not self.session:
            raise ValueError("Session required for database operations")

        nx = self.nx

        # Get rule
        stmt = select(EmbeddingRule).where(EmbeddingRule.rule_id == rule_id)
        rule = self.session.exec(stmt).first()
        if not rule:
            raise ValueError(f"Rule not found: {rule_id}")

        # Convert to graph
        graph = self.rule_to_graph(rule)

        # Compute statistics
        stats = {
            "rule_id": rule_id,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
        }

        # Node type distribution
        node_types: dict[str, int] = {}
        for _, data in graph.nodes(data=True):
            ntype = data.get("node_type", "unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1
        stats["node_types"] = node_types

        # Edge type distribution
        edge_types: dict[str, int] = {}
        for _, _, data in graph.edges(data=True):
            etype = data.get("edge_type", "unknown")
            edge_types[etype] = edge_types.get(etype, 0) + 1
        stats["edge_types"] = edge_types

        # Degree statistics
        degrees = [d for _, d in graph.degree()]
        if degrees:
            stats["degree_stats"] = {
                "mean": sum(degrees) / len(degrees),
                "max": max(degrees),
                "min": min(degrees),
            }
        else:
            stats["degree_stats"] = {"mean": 0, "max": 0, "min": 0}

        # Clustering coefficient
        try:
            stats["avg_clustering"] = nx.average_clustering(graph)
        except Exception:
            stats["avg_clustering"] = 0

        return stats

    def visualize_graph(
        self,
        rule_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Get a visualization-ready representation of a rule's graph.

        Args:
            rule_id: The rule identifier
            format: Output format ('json' for node-link format, 'dot' for Graphviz)

        Returns:
            Graph in requested format

        Raises:
            ValueError: If rule is not found
        """
        from sqlmodel import select

        if not self.session:
            raise ValueError("Session required for database operations")

        nx = self.nx

        # Get rule
        stmt = select(EmbeddingRule).where(EmbeddingRule.rule_id == rule_id)
        rule = self.session.exec(stmt).first()
        if not rule:
            raise ValueError(f"Rule not found: {rule_id}")

        # Convert to graph
        graph = self.rule_to_graph(rule)

        if format == "json":
            # Node-link format for D3.js / vis.js
            return nx.node_link_data(graph)

        elif format == "dot":
            # Graphviz DOT format
            try:
                from networkx.drawing.nx_pydot import to_pydot
                pydot_graph = to_pydot(graph)
                return {"dot": pydot_graph.to_string()}
            except ImportError:
                # Fallback: manual DOT generation
                lines = ["digraph G {"]
                for node, data in graph.nodes(data=True):
                    label = data.get("label", node)
                    ntype = data.get("node_type", "unknown")
                    lines.append(f'  "{node}" [label="{label}" type="{ntype}"];')
                for u, v, data in graph.edges(data=True):
                    etype = data.get("edge_type", "")
                    lines.append(f'  "{u}" -> "{v}" [label="{etype}"];')
                lines.append("}")
                return {"dot": "\n".join(lines)}

        else:
            raise ValueError(f"Unsupported format: {format}")

    def batch_generate_embeddings(
        self,
        rule_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Generate graph embeddings for multiple rules.

        Args:
            rule_ids: List of rule IDs (None = all rules)

        Returns:
            Dict with counts of processed/failed rules
        """
        from sqlmodel import select
        from datetime import datetime, timezone

        if not self.session:
            raise ValueError("Session required for database operations")

        np = self.np
        nx = self.nx

        # Get rules to process
        if rule_ids:
            stmt = select(EmbeddingRule).where(EmbeddingRule.rule_id.in_(rule_ids))
        else:
            stmt = select(EmbeddingRule)

        rules = self.session.exec(stmt).all()

        processed = 0
        failed = 0

        for rule in rules:
            try:
                # Convert to graph
                graph = self.rule_to_graph(rule)

                # Generate embedding
                embedding = self.generate_graph_embedding(graph)

                # Store in database
                graph_json = json.dumps(nx.node_link_data(graph))

                graph_emb = GraphEmbedding(
                    rule_id=rule.id,
                    embedding_vector=embedding.astype(np.float32).tobytes(),
                    vector_json=json.dumps(embedding.tolist()),
                    vector_dim=len(embedding),
                    graph_json=graph_json,
                    num_nodes=graph.number_of_nodes(),
                    num_edges=graph.number_of_edges(),
                    created_at=datetime.now(timezone.utc),
                )

                # Check if embedding already exists
                existing_stmt = select(GraphEmbedding).where(
                    GraphEmbedding.rule_id == rule.id
                )
                existing = self.session.exec(existing_stmt).first()

                if existing:
                    # Update existing
                    existing.embedding_vector = graph_emb.embedding_vector
                    existing.vector_json = graph_emb.vector_json
                    existing.vector_dim = graph_emb.vector_dim
                    existing.graph_json = graph_emb.graph_json
                    existing.num_nodes = graph_emb.num_nodes
                    existing.num_edges = graph_emb.num_edges
                    self.session.add(existing)
                else:
                    # Insert new
                    self.session.add(graph_emb)

                processed += 1

            except Exception:
                failed += 1
                continue

        self.session.commit()

        return {
            "processed": processed,
            "failed": failed,
            "total": len(rules),
        }

    def _cosine_similarity(self, a: Any, b: Any) -> float:
        """Compute cosine similarity between two vectors."""
        np = self.np

        if len(a) != len(b):
            return 0.0

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
