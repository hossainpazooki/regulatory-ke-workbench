"""
Graph visualization components for Streamlit.

Provides reusable UI components for:
- Interactive network graphs (PyVis)
- Graph statistics display
- View mode and filter controls
- Graph comparison
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import streamlit as st
import streamlit.components.v1 as components

# Try to import pyvis (optional)
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# Try to import networkx (optional)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

if TYPE_CHECKING:
    pass


# =============================================================================
# Graph View Controls
# =============================================================================


def render_graph_controls(key_prefix: str = "graph") -> dict:
    """Render controls for graph visualization.

    Args:
        key_prefix: Prefix for widget keys

    Returns:
        Dict with view_mode, depth, node_types
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        view_mode = st.radio(
            "View Mode",
            options=["single_rule", "rule_network", "entity_graph"],
            format_func=lambda x: {
                "single_rule": "Single Rule",
                "rule_network": "Rule Network",
                "entity_graph": "Entity Graph",
            }.get(x, x),
            key=f"{key_prefix}_view_mode",
        )

    with col2:
        depth = st.selectbox(
            "Depth",
            options=[1, 2, 3],
            index=0,
            key=f"{key_prefix}_depth",
            help="How many hops from the selected node",
        )

    with col3:
        node_types = st.multiselect(
            "Show Node Types",
            options=["conditions", "decisions", "legal_refs", "entities"],
            default=["conditions", "decisions", "legal_refs"],
            key=f"{key_prefix}_node_types",
        )

    return {
        "view_mode": view_mode,
        "depth": depth,
        "node_types": node_types,
    }


# =============================================================================
# PyVis Network Graph
# =============================================================================


def render_pyvis_graph(
    nodes: list[dict],
    edges: list[dict],
    height: int = 500,
    physics: bool = True,
    key: str = "pyvis_graph",
) -> None:
    """Render interactive PyVis network graph.

    Args:
        nodes: List of node dicts with id, label, color, etc.
        edges: List of edge dicts with from, to, etc.
        height: Height in pixels
        physics: Whether to enable physics simulation
        key: Widget key
    """
    if not PYVIS_AVAILABLE:
        st.warning("Install pyvis for interactive graphs: pip install pyvis")
        _render_fallback_graph(nodes, edges)
        return

    if not nodes:
        st.info("No graph data to display.")
        return

    # Create PyVis network
    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )

    # Configure physics
    if physics:
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100
                },
                "stabilization": {
                    "iterations": 100
                }
            },
            "nodes": {
                "shape": "dot",
                "size": 16
            },
            "edges": {
                "smooth": {
                    "type": "continuous"
                }
            }
        }
        """)
    else:
        net.toggle_physics(False)

    # Add nodes
    for node in nodes:
        node_id = node.get("id", str(hash(str(node))))
        label = node.get("label", node_id)
        color = node.get("color", "#97c2fc")
        title = node.get("title", label)
        size = node.get("size", 25)
        shape = node.get("shape", "dot")

        net.add_node(
            node_id,
            label=label,
            color=color,
            title=title,
            size=size,
            shape=shape,
        )

    # Add edges
    for edge in edges:
        from_node = edge.get("from") or edge.get("source")
        to_node = edge.get("to") or edge.get("target")
        label = edge.get("label", "")
        color = edge.get("color", "#848484")
        width = edge.get("width", 1)

        if from_node and to_node:
            net.add_edge(
                from_node,
                to_node,
                label=label,
                color=color,
                width=width,
            )

    # Generate HTML and display
    try:
        html = net.generate_html()
        components.html(html, height=height + 50, scrolling=True)
    except Exception as e:
        st.error(f"Error rendering graph: {e}")
        _render_fallback_graph(nodes, edges)


def _render_fallback_graph(nodes: list[dict], edges: list[dict]) -> None:
    """Fallback rendering when PyVis is not available."""
    st.markdown("**Graph Nodes:**")

    if not nodes:
        st.info("No nodes to display.")
        return

    # Show as table
    import pandas as pd

    nodes_df = pd.DataFrame(nodes)
    display_cols = [c for c in ["id", "label", "type", "color"] if c in nodes_df.columns]
    st.dataframe(nodes_df[display_cols] if display_cols else nodes_df, use_container_width=True)

    st.markdown("**Graph Edges:**")
    if edges:
        edges_df = pd.DataFrame(edges)
        st.dataframe(edges_df, use_container_width=True)


# =============================================================================
# Graph Statistics
# =============================================================================


def render_graph_stats(stats: dict) -> None:
    """Display graph statistics in metrics row.

    Args:
        stats: Dict with num_nodes, num_edges, density, clustering, etc.
    """
    cols = st.columns(4)

    with cols[0]:
        st.metric("Nodes", stats.get("num_nodes", 0))

    with cols[1]:
        st.metric("Edges", stats.get("num_edges", 0))

    with cols[2]:
        density = stats.get("density", 0)
        st.metric("Density", f"{density:.2f}")

    with cols[3]:
        clustering = stats.get("clustering", 0)
        st.metric("Clustering", f"{clustering:.2f}")


def compute_graph_stats(nodes: list[dict], edges: list[dict]) -> dict:
    """Compute basic graph statistics.

    Args:
        nodes: List of node dicts
        edges: List of edge dicts

    Returns:
        Dict with num_nodes, num_edges, density, clustering
    """
    num_nodes = len(nodes)
    num_edges = len(edges)

    # Compute density
    max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
    density = num_edges / max_edges if max_edges > 0 else 0

    # Try to compute clustering coefficient if networkx available
    clustering = 0.0
    if NETWORKX_AVAILABLE and num_nodes > 0:
        try:
            G = nx.Graph()
            for node in nodes:
                G.add_node(node.get("id"))
            for edge in edges:
                from_node = edge.get("from") or edge.get("source")
                to_node = edge.get("to") or edge.get("target")
                if from_node and to_node:
                    G.add_edge(from_node, to_node)
            clustering = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
        except Exception:
            clustering = 0.0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "clustering": clustering,
    }


# =============================================================================
# Node Type Colors
# =============================================================================


def get_node_colors() -> dict[str, str]:
    """Get standard colors for different node types.

    Returns:
        Dict mapping node type to color
    """
    return {
        "rule": "#4B8BBE",        # Blue
        "condition": "#FFE873",   # Yellow
        "decision": "#306998",    # Dark blue
        "entity": "#FFD43B",      # Gold
        "legal_ref": "#E34C26",   # Orange
        "jurisdiction": "#6DB33F", # Green
        "default": "#97c2fc",     # Light blue
    }


def get_node_color(node_type: str) -> str:
    """Get color for a specific node type.

    Args:
        node_type: Type of node

    Returns:
        Color hex code
    """
    colors = get_node_colors()
    return colors.get(node_type, colors["default"])


# =============================================================================
# Rule Selection for Graph
# =============================================================================


def render_rule_graph_selector(
    rules: list[dict],
    key: str = "graph_rule_select",
) -> str | None:
    """Render dropdown to select a rule for graph visualization.

    Args:
        rules: List of rule dicts with rule_id
        key: Widget key

    Returns:
        Selected rule_id or None
    """
    if not rules:
        st.warning("No rules available.")
        return None

    rule_ids = [r.get("rule_id", "") for r in rules if r.get("rule_id")]

    return st.selectbox(
        "Select Rule",
        options=rule_ids,
        key=key,
        placeholder="Choose a rule to visualize...",
    )


# =============================================================================
# Graph Comparison
# =============================================================================


def render_graph_comparison(
    graph1: dict,
    graph2: dict,
    height: int = 400,
) -> None:
    """Render two graphs side by side for comparison.

    Args:
        graph1: Dict with nodes and edges for first graph
        graph2: Dict with nodes and edges for second graph
        height: Height per graph
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{graph1.get('title', 'Graph 1')}**")
        render_pyvis_graph(
            graph1.get("nodes", []),
            graph1.get("edges", []),
            height=height,
            key="compare_graph_1",
        )
        render_graph_stats(compute_graph_stats(
            graph1.get("nodes", []),
            graph1.get("edges", []),
        ))

    with col2:
        st.markdown(f"**{graph2.get('title', 'Graph 2')}**")
        render_pyvis_graph(
            graph2.get("nodes", []),
            graph2.get("edges", []),
            height=height,
            key="compare_graph_2",
        )
        render_graph_stats(compute_graph_stats(
            graph2.get("nodes", []),
            graph2.get("edges", []),
        ))


# =============================================================================
# Build Graph from Rule
# =============================================================================


def build_rule_graph(rule: dict) -> tuple[list[dict], list[dict]]:
    """Build graph nodes and edges from a rule.

    Args:
        rule: Rule dict with rule_id, decision_tree, source, etc.

    Returns:
        Tuple of (nodes, edges)
    """
    nodes = []
    edges = []
    colors = get_node_colors()

    rule_id = rule.get("rule_id", "Unknown")

    # Root node (the rule)
    nodes.append({
        "id": rule_id,
        "label": rule_id,
        "type": "rule",
        "color": colors["rule"],
        "size": 30,
        "title": rule.get("description", rule_id),
    })

    # Legal reference node
    source = rule.get("source", {})
    if source:
        doc_id = source.get("document_id", "")
        article = source.get("article", "")
        ref_id = f"{doc_id}_{article}"
        ref_label = f"{doc_id} {article}"

        nodes.append({
            "id": ref_id,
            "label": ref_label,
            "type": "legal_ref",
            "color": colors["legal_ref"],
            "title": f"Legal source: {ref_label}",
        })
        edges.append({
            "from": rule_id,
            "to": ref_id,
            "label": "source",
        })

    # Tags as entity nodes
    tags = rule.get("tags", [])
    for tag in tags[:5]:  # Limit to 5 tags
        tag_id = f"tag_{tag}"
        nodes.append({
            "id": tag_id,
            "label": tag,
            "type": "entity",
            "color": colors["entity"],
            "title": f"Tag: {tag}",
        })
        edges.append({
            "from": rule_id,
            "to": tag_id,
            "label": "tagged",
        })

    # Jurisdiction
    jurisdiction = rule.get("jurisdiction")
    if jurisdiction:
        jur_id = f"jur_{jurisdiction}"
        nodes.append({
            "id": jur_id,
            "label": str(jurisdiction),
            "type": "jurisdiction",
            "color": colors["jurisdiction"],
            "title": f"Jurisdiction: {jurisdiction}",
        })
        edges.append({
            "from": rule_id,
            "to": jur_id,
            "label": "applies_in",
        })

    return nodes, edges


def build_rule_network(
    rules: list[dict],
    similarity_threshold: float = 0.5,
    similarities: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build network of rules connected by similarity.

    Args:
        rules: List of rule dicts
        similarity_threshold: Minimum similarity to create edge
        similarities: Optional pre-computed similarity pairs

    Returns:
        Tuple of (nodes, edges)
    """
    nodes = []
    edges = []
    colors = get_node_colors()

    # Create nodes for each rule
    for rule in rules:
        rule_id = rule.get("rule_id", "")
        jurisdiction = rule.get("jurisdiction", "")

        nodes.append({
            "id": rule_id,
            "label": rule_id,
            "type": "rule",
            "color": colors.get(str(jurisdiction), colors["rule"]),
            "title": rule.get("description", rule_id),
            "jurisdiction": jurisdiction,
        })

    # Create edges from similarities
    if similarities:
        for sim in similarities:
            score = sim.get("score", 0)
            if score >= similarity_threshold:
                edges.append({
                    "from": sim.get("rule1_id"),
                    "to": sim.get("rule2_id"),
                    "label": f"{score:.2f}",
                    "width": 1 + (score * 3),
                })

    return nodes, edges
