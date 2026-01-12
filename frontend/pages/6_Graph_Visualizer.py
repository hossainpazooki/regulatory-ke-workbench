"""
Graph Visualizer - Explore rule relationships and graph structure.

This page provides:
- Single rule graph (conditions -> decisions -> sources)
- Rule network (rules connected by shared entities/sources)
- Interactive PyVis network graph
- Graph statistics display
- Compare two rule graphs side-by-side
"""

import streamlit as st

from frontend.helpers import get_analytics_client
from frontend.ui import (
    render_graph_controls,
    render_pyvis_graph,
    render_graph_stats,
    compute_graph_stats,
    render_rule_graph_selector,
    render_graph_comparison,
    build_rule_graph,
    build_rule_network,
)


# Page config
st.set_page_config(
    page_title="Graph Visualizer",
    page_icon="",
    layout="wide",
)


def main():
    """Main page content."""
    st.title("Graph Visualizer")
    st.markdown(
        "Explore rule relationships and structure through interactive graphs."
    )

    # Initialize client
    client = get_analytics_client()

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # View mode
        view_mode = st.radio(
            "View Mode",
            options=["single_rule", "rule_network", "comparison"],
            format_func=lambda x: {
                "single_rule": "Single Rule",
                "rule_network": "Rule Network",
                "comparison": "Compare Rules",
            }.get(x, x),
            key="view_mode",
        )

        st.markdown("---")

        # Rule selection
        st.subheader("Select Rules")

        # Get available rules from coverage
        try:
            coverage = client.get_coverage()
            available_rules = []
            for framework, data in coverage.get("coverage_by_framework", {}).items():
                for article in data.get("rules_per_article", {}).keys():
                    rule_id = f"{framework.lower()}_{article.lower().replace(' ', '_')}"
                    available_rules.append({
                        "rule_id": rule_id,
                        "framework": framework,
                        "article": article,
                    })
        except Exception:
            available_rules = []

        if view_mode == "single_rule":
            selected_rule = st.text_input(
                "Rule ID",
                key="rule_id_input",
                placeholder="e.g., mica_art36_authorization",
            )
        elif view_mode == "comparison":
            col1, col2 = st.columns(2)
            with col1:
                rule1 = st.text_input(
                    "Rule 1",
                    key="rule1_input",
                    placeholder="First rule",
                )
            with col2:
                rule2 = st.text_input(
                    "Rule 2",
                    key="rule2_input",
                    placeholder="Second rule",
                )
        else:
            selected_rule = None

        # Graph options
        st.markdown("---")
        st.subheader("Options")

        show_physics = st.checkbox(
            "Enable Physics",
            value=True,
            key="show_physics",
            help="Enable force-directed layout",
        )

        node_types = st.multiselect(
            "Node Types",
            options=["conditions", "decisions", "legal_refs", "entities", "jurisdiction"],
            default=["conditions", "decisions", "legal_refs"],
            key="node_types",
        )

        if view_mode == "rule_network":
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key="sim_threshold",
                help="Min similarity to draw edge",
            )

    # Main content
    if view_mode == "single_rule":
        _render_single_rule_graph(client, show_physics)
    elif view_mode == "rule_network":
        _render_rule_network(client, show_physics, similarity_threshold)
    elif view_mode == "comparison":
        _render_comparison(client, rule1, rule2, show_physics)


def _render_single_rule_graph(client, physics: bool):
    """Render single rule graph view."""
    rule_id = st.session_state.get("rule_id_input", "")

    if not rule_id:
        st.info("Enter a rule ID in the sidebar to visualize its structure.")
        return

    st.subheader(f"Rule Graph: `{rule_id}`")

    # Build mock rule data for visualization
    # In production, this would fetch from the API
    rule_data = {
        "rule_id": rule_id,
        "description": f"Visualization of {rule_id}",
        "source": {
            "document_id": rule_id.split("_")[0].upper() if "_" in rule_id else "Unknown",
            "article": " ".join(rule_id.split("_")[1:3]) if "_" in rule_id else "Unknown",
        },
        "tags": ["crypto", "authorization", "compliance"],
        "jurisdiction": "EU" if "mica" in rule_id.lower() else "UK",
    }

    # Build graph
    nodes, edges = build_rule_graph(rule_data)

    # Display stats
    stats = compute_graph_stats(nodes, edges)
    render_graph_stats(stats)

    # Render graph
    render_pyvis_graph(
        nodes=nodes,
        edges=edges,
        height=500,
        physics=physics,
        key="single_rule_graph",
    )

    # Rule details
    with st.expander("Rule Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**ID:** `{rule_id}`")
            st.markdown(f"**Source:** {rule_data['source']['document_id']} {rule_data['source']['article']}")
        with col2:
            st.markdown(f"**Jurisdiction:** {rule_data['jurisdiction']}")
            st.markdown(f"**Tags:** {', '.join(rule_data['tags'])}")


def _render_rule_network(client, physics: bool, threshold: float):
    """Render rule network view."""
    st.subheader("Rule Network")

    with st.spinner("Building rule network..."):
        try:
            # Get coverage to find rules
            coverage = client.get_coverage()
            rules = []
            for framework, data in coverage.get("coverage_by_framework", {}).items():
                for article in data.get("rules_per_article", {}).keys():
                    rules.append({
                        "rule_id": f"{framework.lower()}_{article.lower().replace(' ', '_')}",
                        "jurisdiction": "EU" if framework.upper() == "MICA" else "UK",
                        "description": f"{framework} {article}",
                    })

            if not rules:
                st.info("No rules found in database.")
                return

            # Build network (without pre-computed similarities for now)
            nodes, edges = build_rule_network(
                rules=rules,
                similarity_threshold=threshold,
                similarities=None,  # Would need API endpoint
            )

            # Display stats
            stats = compute_graph_stats(nodes, edges)
            render_graph_stats(stats)

            st.caption(f"Showing {len(rules)} rules with similarity threshold {threshold}")

            # Render graph
            render_pyvis_graph(
                nodes=nodes,
                edges=edges,
                height=600,
                physics=physics,
                key="rule_network_graph",
            )

        except Exception as e:
            st.error(f"Error building network: {e}")


def _render_comparison(client, rule1: str, rule2: str, physics: bool):
    """Render comparison view."""
    st.subheader("Rule Comparison")

    if not rule1 or not rule2:
        st.info("Enter two rule IDs in the sidebar to compare their structures.")
        return

    # Build graphs for both rules
    rule1_data = {
        "rule_id": rule1,
        "description": f"Rule: {rule1}",
        "source": {"document_id": rule1.split("_")[0].upper() if "_" in rule1 else "Unknown", "article": ""},
        "tags": ["tag1", "tag2"],
        "jurisdiction": "EU" if "mica" in rule1.lower() else "UK",
    }

    rule2_data = {
        "rule_id": rule2,
        "description": f"Rule: {rule2}",
        "source": {"document_id": rule2.split("_")[0].upper() if "_" in rule2 else "Unknown", "article": ""},
        "tags": ["tag1", "tag3"],
        "jurisdiction": "UK" if "fca" in rule2.lower() else "EU",
    }

    nodes1, edges1 = build_rule_graph(rule1_data)
    nodes2, edges2 = build_rule_graph(rule2_data)

    graph1 = {
        "title": rule1,
        "nodes": nodes1,
        "edges": edges1,
    }

    graph2 = {
        "title": rule2,
        "nodes": nodes2,
        "edges": edges2,
    }

    render_graph_comparison(graph1, graph2, height=400)

    # Comparison metrics
    st.markdown("---")
    st.subheader("Comparison")

    try:
        comparison = client.compare_rules(rule1, rule2)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Overall Similarity",
                f"{comparison.get('overall_similarity', 0):.2f}",
            )
        with col2:
            shared = comparison.get("shared_entities", [])
            st.metric("Shared Entities", len(shared))
        with col3:
            legal = comparison.get("shared_legal_sources", [])
            st.metric("Shared Sources", len(legal))

        # Similarity by type
        st.markdown("**Similarity by Type:**")
        sim_by_type = comparison.get("similarity_by_type", {})
        cols = st.columns(4)
        for col, (type_name, score) in zip(cols, sim_by_type.items()):
            with col:
                st.metric(type_name.title(), f"{score:.2f}")

    except Exception as e:
        st.warning(f"Could not fetch comparison: {e}")


if __name__ == "__main__":
    main()
