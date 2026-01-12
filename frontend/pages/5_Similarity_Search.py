"""
Similarity Search - Find similar rules using multiple embedding types.

This page provides:
- Search by rule ID with adjustable embedding weights
- Results with similarity breakdown by type
- Natural language explanations of why rules are similar
- Export results to CSV/JSON
"""

import streamlit as st

from frontend.helpers import get_analytics_client
from frontend.ui import (
    render_embedding_type_selector,
    render_search_mode_selector,
    render_weight_sliders,
    render_search_params,
    render_similarity_result,
    render_similarity_results,
    render_export_buttons,
    render_rule_selector,
)


# Page config
st.set_page_config(
    page_title="Similarity Search",
    page_icon="",
    layout="wide",
)


def main():
    """Main page content."""
    st.title("Similarity Search")
    st.markdown(
        "Find similar rules using semantic, structural, entity, and legal embeddings."
    )

    # Initialize client
    client = get_analytics_client()

    # Get available rules
    try:
        summary = client.get_summary()
        total_rules = summary.get("total_rules", 0)
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        total_rules = 0

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Query")

        # Search mode
        search_mode = render_search_mode_selector(key="search_mode")

        st.markdown("---")

        if search_mode == "by_rule":
            # Rule selection
            try:
                # Get rules from coverage endpoint (has rule list)
                coverage = client.get_coverage()
                # Build rule list from coverage data
                rules_list = []
                for framework, data in coverage.get("coverage_by_framework", {}).items():
                    for article, count in data.get("rules_per_article", {}).items():
                        rules_list.append({"rule_id": f"{framework}_{article}"})

                # Actually get rules from the summary endpoint or another way
                # For now, let's use a simple approach
                st.text_input(
                    "Rule ID",
                    key="rule_id_input",
                    placeholder="Enter rule ID (e.g., mica_art36_authorization)",
                )
                query_rule_id = st.session_state.get("rule_id_input", "")

            except Exception:
                query_rule_id = st.text_input(
                    "Rule ID",
                    key="rule_id_input_fallback",
                    placeholder="Enter rule ID",
                )
        else:
            st.info(f"Search mode '{search_mode}' not yet implemented.")
            query_rule_id = ""

        # Embedding type
        st.markdown("---")
        st.subheader("Embedding Type")
        embedding_type = render_embedding_type_selector(
            key="emb_type",
            default="all",
            include_all=True,
            horizontal=False,
        )

        # Weights (only shown for 'all' type)
        if embedding_type == "all":
            st.markdown("---")
            st.subheader("Embedding Weights")
            weights = render_weight_sliders(key_prefix="weight")
        else:
            weights = None

        # Search parameters
        st.markdown("---")
        st.subheader("Parameters")
        search_params = render_search_params(key_prefix="search")

        # Search button
        st.markdown("---")
        search_clicked = st.button(
            "Search",
            type="primary",
            use_container_width=True,
        )

    with col2:
        st.header("Results")

        # Perform search
        if search_clicked and query_rule_id:
            with st.spinner("Searching for similar rules..."):
                try:
                    results = client.get_similar_rules(
                        rule_id=query_rule_id,
                        embedding_type=embedding_type,
                        top_k=search_params["top_k"],
                        min_score=search_params["min_score"],
                        include_explanation=True,
                    )

                    # Store results in session state
                    st.session_state.search_results = results
                    st.session_state.search_query = query_rule_id

                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state.search_results = None

        # Display results
        results = st.session_state.get("search_results")
        query = st.session_state.get("search_query", "")

        if results:
            similar_rules = results.get("similar_rules", [])

            # Results header
            st.markdown(f"**Query:** `{results.get('query_rule_id', query)}`")
            if results.get("query_rule_name"):
                st.caption(results.get("query_rule_name"))

            st.markdown(f"**Found:** {len(similar_rules)} similar rules")

            st.markdown("---")

            # Render result cards
            if similar_rules:
                for i, rule in enumerate(similar_rules):
                    render_similarity_result(
                        result=rule,
                        expanded=(i == 0),  # First result expanded
                        key=f"result_{i}",
                    )

                # Export buttons
                st.markdown("---")
                render_export_buttons(
                    data=similar_rules,
                    filename_prefix=f"similar_to_{query}",
                )
            else:
                st.info(
                    "No similar rules found. "
                    "Try lowering the minimum score threshold."
                )

        elif query_rule_id and not search_clicked:
            st.info("Click 'Search' to find similar rules.")
        else:
            st.info("Enter a rule ID and click 'Search' to find similar rules.")

        # Quick stats
        if total_rules > 0:
            st.markdown("---")
            st.caption(f"Database: {total_rules} rules available")


if __name__ == "__main__":
    main()
