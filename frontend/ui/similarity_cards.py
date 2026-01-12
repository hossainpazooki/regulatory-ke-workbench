"""
Similarity search result card components for Streamlit.

Provides reusable UI components for:
- Similarity search result cards
- Weight sliders for embedding types
- Search mode selection
- Explanation rendering
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    pass


# =============================================================================
# Search Mode Selection
# =============================================================================


def render_search_mode_selector(key: str = "search_mode") -> str:
    """Render radio buttons for search mode selection.

    Args:
        key: Streamlit widget key

    Returns:
        Selected search mode (by_rule, by_text, by_entities)
    """
    modes = {
        "by_rule": "By Rule",
        "by_text": "By Text",
        "by_entities": "By Entities",
    }

    return st.radio(
        "Search Mode",
        options=list(modes.keys()),
        format_func=lambda x: modes[x],
        key=key,
        horizontal=True,
    )


# =============================================================================
# Weight Sliders
# =============================================================================


def render_weight_sliders(key_prefix: str = "weight") -> dict[str, float]:
    """Render sliders for embedding type weights.

    Args:
        key_prefix: Prefix for widget keys

    Returns:
        Dict mapping embedding type to weight (normalized)
    """
    st.markdown("**Embedding Weights:**")

    col1, col2 = st.columns(2)

    with col1:
        semantic = st.slider(
            "Semantic",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            key=f"{key_prefix}_semantic",
        )
        structural = st.slider(
            "Structural",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            key=f"{key_prefix}_structural",
        )

    with col2:
        entity = st.slider(
            "Entity",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            key=f"{key_prefix}_entity",
        )
        legal = st.slider(
            "Legal",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            key=f"{key_prefix}_legal",
        )

    # Normalize weights
    total = semantic + structural + entity + legal
    if total > 0:
        return {
            "semantic": semantic / total,
            "structural": structural / total,
            "entity": entity / total,
            "legal": legal / total,
        }
    return {
        "semantic": 0.25,
        "structural": 0.25,
        "entity": 0.25,
        "legal": 0.25,
    }


# =============================================================================
# Search Parameters
# =============================================================================


def render_search_params(key_prefix: str = "search") -> dict:
    """Render search parameter inputs.

    Args:
        key_prefix: Prefix for widget keys

    Returns:
        Dict with top_k and min_score
    """
    col1, col2 = st.columns(2)

    with col1:
        top_k = st.number_input(
            "Max Results",
            min_value=1,
            max_value=50,
            value=10,
            key=f"{key_prefix}_top_k",
        )

    with col2:
        min_score = st.slider(
            "Min Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=f"{key_prefix}_min_score",
        )

    return {
        "top_k": top_k,
        "min_score": min_score,
    }


# =============================================================================
# Similarity Result Card
# =============================================================================


def render_similarity_result(
    result: dict,
    expanded: bool = False,
    key: str | None = None,
) -> bool:
    """Render a single similarity search result card.

    Args:
        result: Dict with rule_id, name, overall_score, scores_by_type, explanation
        expanded: Whether to show expanded details
        key: Widget key for expansion state

    Returns:
        True if card was clicked/expanded
    """
    rule_id = result.get("rule_id", "Unknown")
    rule_name = result.get("rule_name") or result.get("name", "")
    overall_score = result.get("overall_score", 0)
    jurisdiction = result.get("jurisdiction", "N/A")
    scores_by_type = result.get("scores_by_type", {})
    explanation = result.get("explanation", {})

    # Create header with score badge
    score_color = _score_to_color(overall_score)
    header = f"**{rule_id}** :gray[({jurisdiction})]"

    with st.expander(header, expanded=expanded):
        # Score row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Overall", f"{overall_score:.2f}")

        with col2:
            st.metric("Semantic", f"{scores_by_type.get('semantic', 0):.2f}")

        with col3:
            st.metric("Entity", f"{scores_by_type.get('entity', 0):.2f}")

        with col4:
            st.metric("Legal", f"{scores_by_type.get('legal', 0):.2f}")

        # Description
        if rule_name:
            st.caption(rule_name)

        # Explanation section
        if explanation:
            _render_explanation(explanation)

        return True

    return False


def _score_to_color(score: float) -> str:
    """Convert similarity score to color."""
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "orange"
    else:
        return "red"


def _render_explanation(explanation: dict) -> None:
    """Render similarity explanation details."""
    st.markdown("---")
    st.markdown("**Why Similar:**")

    primary = explanation.get("primary_reason", "")
    if primary:
        st.markdown(f"• {primary}")

    shared_entities = explanation.get("shared_entities", [])
    if shared_entities:
        st.markdown(f"• Shared entities: `{', '.join(shared_entities[:5])}`")

    shared_legal = explanation.get("shared_legal_sources", [])
    if shared_legal:
        st.markdown(f"• Shared legal sources: `{', '.join(shared_legal[:3])}`")

    structural = explanation.get("structural_similarity", "")
    if structural:
        st.markdown(f"• Structure: {structural}")


# =============================================================================
# Results List
# =============================================================================


def render_similarity_results(
    results: list[dict],
    on_select: callable | None = None,
) -> str | None:
    """Render a list of similarity search results.

    Args:
        results: List of result dicts
        on_select: Optional callback when a result is selected

    Returns:
        Selected rule_id or None
    """
    if not results:
        st.info("No similar rules found. Try lowering the minimum score threshold.")
        return None

    selected = None

    for i, result in enumerate(results):
        rule_id = result.get("rule_id", f"Unknown_{i}")
        if render_similarity_result(result, expanded=False, key=f"result_{i}"):
            selected = rule_id

    return selected


# =============================================================================
# Score Bar
# =============================================================================


def render_score_bar(
    scores: dict[str, float],
    key: str = "score_bar",
) -> None:
    """Render a horizontal bar showing score breakdown by type.

    Args:
        scores: Dict mapping embedding type to score
        key: Widget key
    """
    if not scores:
        return

    # Create columns for each score type
    cols = st.columns(len(scores))

    colors = {
        "semantic": "blue",
        "structural": "green",
        "entity": "orange",
        "legal": "purple",
    }

    for col, (type_name, score) in zip(cols, scores.items()):
        with col:
            pct = int(score * 100)
            color = colors.get(type_name, "gray")
            st.markdown(f"**{type_name.title()}**")
            st.progress(score, text=f"{pct}%")


# =============================================================================
# Export Buttons
# =============================================================================


def render_export_buttons(
    data: list[dict],
    filename_prefix: str = "similarity_results",
) -> None:
    """Render buttons to export results as CSV/JSON.

    Args:
        data: List of result dicts to export
        filename_prefix: Prefix for downloaded file
    """
    import json

    col1, col2 = st.columns(2)

    with col1:
        # CSV export
        if data:
            import pandas as pd
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"{filename_prefix}.csv",
                mime="text/csv",
            )

    with col2:
        # JSON export
        if data:
            json_str = json.dumps(data, indent=2)
            st.download_button(
                "Download JSON",
                data=json_str,
                file_name=f"{filename_prefix}.json",
                mime="application/json",
            )


# =============================================================================
# Query Input
# =============================================================================


def render_rule_selector(
    rules: list[dict],
    key: str = "rule_select",
) -> str | None:
    """Render a dropdown to select a rule for similarity search.

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

    if not rule_ids:
        st.warning("No valid rule IDs found.")
        return None

    return st.selectbox(
        "Select Rule",
        options=rule_ids,
        key=key,
        placeholder="Choose a rule...",
    )


def render_text_query_input(
    key: str = "text_query",
    placeholder: str = "Enter description or keywords...",
) -> str:
    """Render text input for free-text similarity search.

    Args:
        key: Widget key
        placeholder: Placeholder text

    Returns:
        Entered text
    """
    return st.text_area(
        "Query Text",
        placeholder=placeholder,
        key=key,
        height=100,
    )


def render_entity_selector(
    available_entities: list[str],
    key: str = "entity_select",
) -> list[str]:
    """Render multiselect for entity-based search.

    Args:
        available_entities: List of available entity names
        key: Widget key

    Returns:
        List of selected entities
    """
    return st.multiselect(
        "Select Entities",
        options=available_entities,
        key=key,
        placeholder="Choose entities to search for...",
    )
