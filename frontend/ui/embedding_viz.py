"""
Embedding visualization components for Streamlit.

Provides reusable UI components for:
- UMAP 2D/3D scatter plots
- Embedding type selection
- UMAP parameter controls
- Point hover/click interactions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

# Try to import plotly (optional but recommended)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

if TYPE_CHECKING:
    pass


# =============================================================================
# Embedding Type Selection
# =============================================================================


def render_embedding_type_selector(
    key: str = "embedding_type",
    default: str = "semantic",
    include_all: bool = True,
    horizontal: bool = True,
) -> str:
    """Render radio buttons for embedding type selection.

    Args:
        key: Streamlit widget key
        default: Default selected type
        include_all: Whether to include "all" option
        horizontal: Whether to display horizontally

    Returns:
        Selected embedding type
    """
    options = ["semantic", "structural", "entity", "legal"]
    if include_all:
        options = ["all"] + options

    labels = {
        "all": "All Types",
        "semantic": "Semantic",
        "structural": "Structural",
        "entity": "Entity",
        "legal": "Legal",
    }

    # Find default index
    default_idx = options.index(default) if default in options else 0

    selected = st.radio(
        "Embedding Type",
        options=options,
        format_func=lambda x: labels.get(x, x),
        index=default_idx,
        key=key,
        horizontal=horizontal,
    )

    return selected


# =============================================================================
# UMAP Controls
# =============================================================================


def render_umap_controls(key_prefix: str = "umap") -> dict:
    """Render sliders for UMAP parameters.

    Args:
        key_prefix: Prefix for widget keys

    Returns:
        Dict with n_components, n_neighbors, min_dist
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        n_components = st.selectbox(
            "Dimensions",
            options=[2, 3],
            index=0,
            key=f"{key_prefix}_n_components",
            help="2D or 3D projection",
        )

    with col2:
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=2,
            max_value=50,
            value=15,
            key=f"{key_prefix}_n_neighbors",
            help="Local neighborhood size",
        )

    with col3:
        min_dist = st.slider(
            "min_dist",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            key=f"{key_prefix}_min_dist",
            help="Minimum distance in projection",
        )

    return {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
    }


# =============================================================================
# UMAP Scatter Plot
# =============================================================================


def render_umap_scatter(
    points: list[dict],
    color_by: str = "jurisdiction",
    dimensions: int = 2,
    selected_rule_id: str | None = None,
    height: int = 500,
    key: str = "umap_scatter",
) -> str | None:
    """Render UMAP projection as Plotly scatter.

    Args:
        points: List of dicts with rule_id, x, y, z?, cluster_id?, jurisdiction?, etc.
        color_by: Field to use for coloring points
        dimensions: 2 or 3 dimensions
        selected_rule_id: Highlight this rule if provided
        height: Chart height in pixels
        key: Widget key

    Returns:
        rule_id of clicked point, or None
    """
    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly for scatter plots: pip install plotly")
        _render_fallback_scatter(points)
        return None

    if not points:
        st.info("No points to display. Ensure embeddings are generated.")
        return None

    # Prepare dataframe-like structure
    import pandas as pd

    df = pd.DataFrame(points)

    # Ensure required columns exist
    if "x" not in df.columns or "y" not in df.columns:
        st.error("Points must have 'x' and 'y' coordinates")
        return None

    # Set color column
    if color_by not in df.columns:
        df[color_by] = "default"

    # Create hover text
    df["hover_text"] = df.apply(
        lambda row: f"<b>{row.get('rule_id', 'Unknown')}</b><br>"
                    f"Jurisdiction: {row.get('jurisdiction', 'N/A')}<br>"
                    f"Cluster: {row.get('cluster_id', 'N/A')}",
        axis=1,
    )

    # Highlight selected rule
    df["size"] = df["rule_id"].apply(
        lambda r: 15 if r == selected_rule_id else 8
    )

    if dimensions == 3 and "z" in df.columns:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color=color_by,
            hover_data=["rule_id", "jurisdiction"],
            custom_data=["rule_id"],
            size="size",
            size_max=15,
        )
    else:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=color_by,
            hover_data=["rule_id", "jurisdiction"],
            custom_data=["rule_id"],
            size="size",
            size_max=15,
        )

    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
    )

    # Display and capture clicks
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select="rerun",
        selection_mode="points",
    )

    # Handle point selection
    if event and "selection" in event:
        points_selected = event["selection"].get("points", [])
        if points_selected:
            # Get the rule_id from custom_data
            point = points_selected[0]
            if "customdata" in point and point["customdata"]:
                return point["customdata"][0]

    return None


def _render_fallback_scatter(points: list[dict]) -> None:
    """Fallback rendering when Plotly is not available."""
    st.markdown("**Embedding Points:**")

    if not points:
        st.info("No points to display.")
        return

    # Show as a table
    import pandas as pd

    display_cols = ["rule_id", "x", "y", "jurisdiction", "cluster_id"]
    df = pd.DataFrame(points)
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True)


# =============================================================================
# Color-By Selector
# =============================================================================


def render_color_by_selector(
    options: list[str] | None = None,
    key: str = "color_by",
) -> str:
    """Render dropdown for selecting point coloring.

    Args:
        options: List of field options
        key: Widget key

    Returns:
        Selected field name
    """
    if options is None:
        options = ["jurisdiction", "cluster_id", "embedding_type"]

    labels = {
        "jurisdiction": "Jurisdiction",
        "cluster_id": "Cluster",
        "embedding_type": "Embedding Type",
        "rule_type": "Rule Type",
    }

    return st.selectbox(
        "Color by",
        options=options,
        format_func=lambda x: labels.get(x, x),
        key=key,
    )


# =============================================================================
# Selected Rule Details Panel
# =============================================================================


def render_selected_rule_details(
    rule_id: str,
    details: dict | None = None,
    neighbors: list[dict] | None = None,
) -> None:
    """Render details panel for a selected rule.

    Args:
        rule_id: Selected rule ID
        details: Optional dict with rule metadata
        neighbors: Optional list of nearest neighbor rules
    """
    st.markdown(f"### Selected: `{rule_id}`")

    if details:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jurisdiction", details.get("jurisdiction", "N/A"))
        with col2:
            st.metric("Framework", details.get("framework", "N/A"))
        with col3:
            st.metric("Cluster", details.get("cluster_id", "N/A"))

        if details.get("description"):
            st.caption(details["description"])

    if neighbors:
        st.markdown("**Nearest Neighbors:**")
        for i, n in enumerate(neighbors[:5], 1):
            score = n.get("score", 0)
            n_id = n.get("rule_id", "Unknown")
            st.markdown(f"{i}. `{n_id}` (similarity: {score:.2f})")


# =============================================================================
# Cluster Selection
# =============================================================================


def render_cluster_selector(
    clusters: list[dict],
    key: str = "cluster_select",
) -> int | None:
    """Render dropdown for cluster selection.

    Args:
        clusters: List of cluster dicts with cluster_id, size, etc.
        key: Widget key

    Returns:
        Selected cluster ID or None
    """
    if not clusters:
        st.info("No clusters available.")
        return None

    options = [None] + [c.get("cluster_id") for c in clusters]
    labels = {
        None: "All Clusters",
    }
    for c in clusters:
        cid = c.get("cluster_id")
        size = c.get("size", 0)
        labels[cid] = f"Cluster {cid} ({size} rules)"

    return st.selectbox(
        "Filter by Cluster",
        options=options,
        format_func=lambda x: labels.get(x, f"Cluster {x}"),
        key=key,
    )
