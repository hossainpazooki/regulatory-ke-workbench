"""
Analytics Dashboard - High-level analytics: clusters, coverage, conflicts.

This page provides:
- Clusters tab: UMAP visualization + cluster details + silhouette score
- Coverage tab: Framework coverage matrix + gap identification
- Conflicts tab: Conflict list with severity + resolution hints
"""

import streamlit as st

from frontend.helpers import get_analytics_client
from frontend.ui import (
    render_umap_scatter,
    render_export_buttons,
)


# Page config
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="",
    layout="wide",
)


def main():
    """Main page content."""
    st.title("Analytics Dashboard")
    st.markdown(
        "High-level analytics for rule comparison, clustering, and conflict detection."
    )

    # Initialize client
    client = get_analytics_client()

    # Summary stats at top
    try:
        summary = client.get_summary()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rules", summary.get("total_rules", 0))
        with col2:
            jurisdictions = summary.get("jurisdictions", [])
            st.metric("Jurisdictions", len(jurisdictions))
        with col3:
            frameworks = summary.get("frameworks", [])
            st.metric("Frameworks", len(frameworks))
        with col4:
            emb_types = summary.get("embedding_types_available", [])
            st.metric("Embedding Types", len(emb_types))
    except Exception as e:
        st.error(f"Error fetching summary: {e}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Clusters", "Coverage", "Conflicts"])

    with tab1:
        _render_clusters_tab(client)

    with tab2:
        _render_coverage_tab(client)

    with tab3:
        _render_conflicts_tab(client)


def _render_clusters_tab(client):
    """Render the clusters analysis tab."""
    st.header("Rule Clusters")

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        embedding_type = st.selectbox(
            "Embedding Type",
            options=["semantic", "structural", "entity", "legal"],
            key="cluster_emb_type",
        )

    with col2:
        n_clusters = st.number_input(
            "Number of Clusters",
            min_value=2,
            max_value=20,
            value=5,
            key="n_clusters",
            help="Set to auto-detect optimal number",
        )

    with col3:
        algorithm = st.selectbox(
            "Algorithm",
            options=["kmeans", "dbscan", "hierarchical"],
            key="cluster_algo",
        )

    # Fetch clusters
    if st.button("Compute Clusters", type="primary"):
        with st.spinner("Computing clusters..."):
            try:
                clusters_data = client.get_clusters(
                    embedding_type=embedding_type,
                    n_clusters=n_clusters,
                    algorithm=algorithm,
                )
                st.session_state.clusters_data = clusters_data
            except Exception as e:
                st.error(f"Error computing clusters: {e}")

    # Display clusters
    clusters_data = st.session_state.get("clusters_data")

    if clusters_data:
        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clusters Found", clusters_data.get("num_clusters", 0))
        with col2:
            st.metric("Algorithm", clusters_data.get("algorithm", "N/A"))
        with col3:
            silhouette = clusters_data.get("silhouette_score", 0)
            st.metric("Silhouette Score", f"{silhouette:.3f}")
        with col4:
            st.metric("Total Rules", clusters_data.get("total_rules", 0))

        st.markdown("---")

        # Cluster details
        clusters = clusters_data.get("clusters", [])

        if clusters:
            # Two column layout
            left_col, right_col = st.columns([2, 1])

            with left_col:
                st.subheader("Cluster Visualization")

                # Get UMAP projection for visualization
                try:
                    projection = client.get_umap_projection(
                        embedding_type=embedding_type,
                        n_components=2,
                    )
                    points = projection.get("points", [])

                    # Add cluster IDs to points
                    rule_to_cluster = {}
                    for cluster in clusters:
                        cid = cluster.get("cluster_id", 0)
                        for rid in cluster.get("rule_ids", []):
                            rule_to_cluster[rid] = cid

                    for point in points:
                        rid = point.get("rule_id", "")
                        point["cluster_id"] = rule_to_cluster.get(rid, -1)

                    render_umap_scatter(
                        points=points,
                        color_by="cluster_id",
                        dimensions=2,
                        height=400,
                        key="cluster_scatter",
                    )
                except Exception as e:
                    st.warning(f"Could not render visualization: {e}")

            with right_col:
                st.subheader("Cluster Details")

                for cluster in clusters:
                    cid = cluster.get("cluster_id", 0)
                    size = cluster.get("size", 0)
                    cohesion = cluster.get("cohesion_score", 0)
                    centroid = cluster.get("centroid_rule_id")
                    keywords = cluster.get("keywords", [])[:3]

                    with st.expander(f"Cluster {cid} ({size} rules)", expanded=(cid == 0)):
                        st.markdown(f"**Cohesion:** {cohesion:.3f}")
                        if centroid:
                            st.markdown(f"**Centroid:** `{centroid}`")
                        if keywords:
                            st.markdown(f"**Keywords:** {', '.join(keywords)}")

                        # Show rule list
                        rule_ids = cluster.get("rule_ids", [])
                        if rule_ids:
                            st.markdown("**Rules:**")
                            for rid in rule_ids[:5]:
                                st.markdown(f"- `{rid}`")
                            if len(rule_ids) > 5:
                                st.caption(f"... and {len(rule_ids) - 5} more")

    else:
        st.info("Click 'Compute Clusters' to analyze rule clusters.")


def _render_coverage_tab(client):
    """Render the coverage analysis tab."""
    st.header("Legal Source Coverage")

    # Fetch coverage
    with st.spinner("Analyzing coverage..."):
        try:
            coverage = client.get_coverage()
        except Exception as e:
            st.error(f"Error fetching coverage: {e}")
            return

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rules", coverage.get("total_rules", 0))
    with col2:
        st.metric("Legal Sources", coverage.get("total_legal_sources", 0))
    with col3:
        overall = coverage.get("overall_coverage_percentage", 0)
        st.metric("Overall Coverage", f"{overall:.1f}%")

    st.markdown("---")

    # Framework coverage matrix
    st.subheader("Coverage by Framework")

    frameworks = coverage.get("coverage_by_framework", {})

    if frameworks:
        # Create coverage matrix display
        for framework_name, framework_data in frameworks.items():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 4])

            with col1:
                st.markdown(f"**{framework_name}**")

            with col2:
                covered = framework_data.get("covered_articles", 0)
                total = framework_data.get("total_articles", 0)
                st.caption(f"{covered}/{total} articles")

            with col3:
                pct = framework_data.get("coverage_percentage", 0)
                st.metric("Coverage", f"{pct:.0f}%", label_visibility="collapsed")

            with col4:
                pct = framework_data.get("coverage_percentage", 0)
                st.progress(pct / 100)

            # Show article breakdown in expander
            with st.expander(f"{framework_name} Details"):
                rules_per_article = framework_data.get("rules_per_article", {})
                if rules_per_article:
                    for article, count in rules_per_article.items():
                        st.markdown(f"- **{article}**: {count} rule(s)")
                else:
                    st.caption("No articles covered")

    else:
        st.info("No framework coverage data available.")

    st.markdown("---")

    # Coverage gaps
    st.subheader("Coverage Gaps")

    gaps = coverage.get("coverage_gaps", [])
    uncovered = coverage.get("uncovered_sources", [])

    if gaps:
        for gap in gaps:
            importance = gap.get("importance", "medium")
            icon = {"high": "", "medium": "", "low": ""}.get(importance, "")
            framework = gap.get("framework", "Unknown")
            article = gap.get("article", "Unknown")
            recommendation = gap.get("recommendation", "")

            st.markdown(f"{icon} **{framework} {article}** ({importance} importance)")
            if recommendation:
                st.caption(recommendation)
    elif uncovered:
        st.markdown("**Uncovered sources:**")
        for source in uncovered[:10]:
            st.markdown(f"- {source}")
    else:
        st.success("No coverage gaps identified!")


def _render_conflicts_tab(client):
    """Render the conflicts detection tab."""
    st.header("Rule Conflicts")

    # Controls
    col1, col2 = st.columns(2)

    with col1:
        conflict_types = st.multiselect(
            "Conflict Types",
            options=["semantic", "structural", "jurisdiction"],
            default=["semantic", "structural"],
            key="conflict_types",
        )

    with col2:
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="conflict_threshold",
            help="Higher = more similar rules flagged",
        )

    # Fetch conflicts
    if st.button("Detect Conflicts", type="primary"):
        with st.spinner("Scanning for conflicts..."):
            try:
                conflicts = client.find_conflicts(
                    conflict_types=conflict_types,
                    threshold=threshold,
                )
                st.session_state.conflicts_data = conflicts
            except Exception as e:
                st.error(f"Error detecting conflicts: {e}")

    # Display conflicts
    conflicts_data = st.session_state.get("conflicts_data")

    if conflicts_data:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rules Analyzed", conflicts_data.get("total_rules_analyzed", 0))
        with col2:
            st.metric("Conflicts Found", conflicts_data.get("conflicts_found", 0))
        with col3:
            st.metric("High Severity", conflicts_data.get("high_severity_count", 0))
        with col4:
            st.metric("Medium Severity", conflicts_data.get("medium_severity_count", 0))

        st.markdown("---")

        # Conflict list
        conflicts = conflicts_data.get("conflicts", [])

        if conflicts:
            for conflict in conflicts:
                rule1 = conflict.get("rule1_id", "Unknown")
                rule2 = conflict.get("rule2_id", "Unknown")
                ctype = conflict.get("conflict_type", "unknown")
                severity = conflict.get("severity", "medium")
                description = conflict.get("description", "")
                score = conflict.get("similarity_score", 0)

                # Severity styling
                severity_colors = {
                    "high": "",
                    "medium": "",
                    "low": "",
                }
                icon = severity_colors.get(severity, "")

                with st.expander(
                    f"{icon} {rule1} {rule2} ({ctype}, {severity})",
                    expanded=(severity == "high"),
                ):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Rule 1:** `{rule1}`")
                    with col2:
                        st.markdown(f"**Rule 2:** `{rule2}`")
                    with col3:
                        st.markdown(f"**Similarity:** {score:.2f}")

                    st.markdown(f"**Type:** {ctype}")
                    st.markdown(f"**Severity:** {severity}")

                    if description:
                        st.markdown(f"**Description:** {description}")

                    # Conflicting aspects
                    aspects = conflict.get("conflicting_aspects", [])
                    if aspects:
                        st.markdown("**Conflicting Aspects:**")
                        for aspect in aspects:
                            st.markdown(f"- {aspect}")

                    # Resolution hints
                    hints = conflict.get("resolution_hints", [])
                    if hints:
                        st.markdown("**Resolution Hints:**")
                        for hint in hints:
                            st.markdown(f"- {hint}")

            # Export
            st.markdown("---")
            render_export_buttons(
                data=conflicts,
                filename_prefix="rule_conflicts",
            )

        else:
            st.success("No conflicts detected!")

    else:
        st.info("Click 'Detect Conflicts' to scan for rule conflicts.")


if __name__ == "__main__":
    main()
