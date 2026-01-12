"""
Embedding Explorer - Visualize rule embeddings in 2D/3D space using UMAP.

This page provides:
- Interactive UMAP scatter plot of rule embeddings
- Toggle between embedding types (semantic, structural, entity, legal)
- 2D/3D visualization options
- Point coloring by jurisdiction or cluster
- Click to inspect rule details
"""

import streamlit as st

from frontend.helpers import get_analytics_client
from frontend.ui import (
    render_embedding_type_selector,
    render_umap_controls,
    render_umap_scatter,
    render_color_by_selector,
    render_selected_rule_details,
    render_cluster_selector,
)


# Page config
st.set_page_config(
    page_title="Embedding Explorer",
    page_icon="",
    layout="wide",
)


def main():
    """Main page content."""
    st.title("Embedding Explorer")
    st.markdown(
        "Visualize rule embeddings in 2D/3D space using UMAP dimensionality reduction."
    )

    # Initialize client
    client = get_analytics_client()

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # Embedding type selection
        embedding_type = render_embedding_type_selector(
            key="emb_type",
            default="semantic",
            include_all=False,
        )

        # UMAP parameters
        st.markdown("---")
        st.subheader("UMAP Parameters")
        umap_params = render_umap_controls(key_prefix="umap")

        # Color by selection
        st.markdown("---")
        color_by = render_color_by_selector(
            options=["jurisdiction", "cluster_id"],
            key="color_by",
        )

        # Refresh button
        st.markdown("---")
        if st.button("Refresh Data", key="refresh"):
            st.cache_data.clear()

    # Main content
    col1, col2 = st.columns([3, 1])

    with col1:
        # Fetch UMAP projection
        with st.spinner("Computing UMAP projection..."):
            try:
                projection = client.get_umap_projection(
                    embedding_type=embedding_type,
                    n_components=umap_params["n_components"],
                    n_neighbors=umap_params["n_neighbors"],
                    min_dist=umap_params["min_dist"],
                )
                points = projection.get("points", [])

                # Show stats
                st.caption(
                    f"Showing {len(points)} rules | "
                    f"Embedding: {projection.get('embedding_type', embedding_type)} | "
                    f"Dimensions: {projection.get('n_components', 2)}D"
                )

            except Exception as e:
                st.error(f"Error fetching projection: {e}")
                points = []

        # Get selected rule from session state
        selected_rule = st.session_state.get("selected_rule_id")

        # Render scatter plot
        if points:
            clicked_rule = render_umap_scatter(
                points=points,
                color_by=color_by,
                dimensions=umap_params["n_components"],
                selected_rule_id=selected_rule,
                height=600,
                key="umap_scatter",
            )

            if clicked_rule:
                st.session_state.selected_rule_id = clicked_rule
                st.rerun()
        else:
            st.info(
                "No embeddings available. "
                "Run the embedding service to generate rule embeddings."
            )

    with col2:
        st.subheader("Selected Rule")

        selected_rule = st.session_state.get("selected_rule_id")

        if selected_rule:
            # Get similar rules for neighbors
            try:
                similar = client.get_similar_rules(
                    rule_id=selected_rule,
                    embedding_type=embedding_type,
                    top_k=5,
                    min_score=0.0,
                )
                neighbors = similar.get("similar_rules", [])

                # Find point details
                point_details = None
                for p in points:
                    if p.get("rule_id") == selected_rule:
                        point_details = p
                        break

                render_selected_rule_details(
                    rule_id=selected_rule,
                    details=point_details,
                    neighbors=neighbors,
                )

            except Exception as e:
                st.error(f"Error fetching details: {e}")
                render_selected_rule_details(
                    rule_id=selected_rule,
                    details=None,
                    neighbors=None,
                )
        else:
            st.info("Click on a point in the scatter plot to see details.")

        # Cluster filter
        st.markdown("---")
        st.subheader("Filter by Cluster")

        # Try to get clusters
        try:
            clusters_data = client.get_clusters(
                embedding_type=embedding_type,
                algorithm="kmeans",
            )
            clusters = clusters_data.get("clusters", [])

            selected_cluster = render_cluster_selector(
                clusters=clusters,
                key="cluster_filter",
            )

            if selected_cluster is not None:
                # Filter points by cluster
                filtered_rules = [
                    c.get("rule_ids", [])
                    for c in clusters
                    if c.get("cluster_id") == selected_cluster
                ]
                if filtered_rules:
                    st.info(f"Cluster {selected_cluster}: {len(filtered_rules[0])} rules")

        except Exception:
            st.caption("Cluster filtering unavailable")


if __name__ == "__main__":
    main()
