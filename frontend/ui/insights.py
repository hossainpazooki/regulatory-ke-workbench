"""
Insights module - Chart rendering and tool gallery components.

This module contains reusable functions for:
- Tree structure rendering (fallback when Supertree unavailable)
- Chart rendering with Visual/Data tabs
- Tool gallery for cockpit home screen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import streamlit as st

if TYPE_CHECKING:
    pass


# =============================================================================
# Tree Rendering (Fallback)
# =============================================================================


def render_tree_view(tree_data: dict, depth: int = 0) -> None:
    """Render a tree structure as nested Streamlit expandables.

    This is a fallback renderer when Supertree is not installed.

    Args:
        tree_data: Dict with 'title', 'children', and optional metadata
        depth: Current recursion depth (for indentation)
    """
    title = tree_data.get("title", "Node")
    children = tree_data.get("children", [])

    # Create display label with metadata
    label_parts = [title]
    if "count" in tree_data:
        label_parts.append(f"({tree_data['count']})")
    if "type" in tree_data:
        label_parts.append(f"[{tree_data['type']}]")

    label = " ".join(label_parts)

    if children:
        with st.expander(label, expanded=(depth < 1)):
            # Show additional metadata
            for key, value in tree_data.items():
                if key not in ("title", "children", "count", "type"):
                    if isinstance(value, (str, int, float, bool)):
                        st.caption(f"{key}: {value}")

            for child in children:
                render_tree_view(child, depth + 1)
    else:
        # Leaf node
        st.markdown(f"{'  ' * depth}• **{label}**")
        for key, value in tree_data.items():
            if key not in ("title", "children", "count", "type"):
                if isinstance(value, (str, int, float, bool)):
                    st.caption(f"{'  ' * depth}  {key}: {value}")


def render_chart(
    chart_type: str,
    tree_data: dict,
    html_renderer: Callable[[dict], str],
    height: int = 500,
) -> None:
    """Render a chart with fallback to JSON view.

    Shows Visual tab (Supertree HTML or tree view) and Data tab (JSON).

    Args:
        chart_type: Name of the chart type for display
        tree_data: The tree data structure to render
        html_renderer: Function to render tree_data as HTML
        height: Height of the HTML component in pixels
    """
    from backend.core.visualization import is_supertree_available

    tab1, tab2 = st.tabs(["Visual", "Data"])

    with tab1:
        if is_supertree_available():
            html = html_renderer(tree_data)
            st.components.v1.html(html, height=height, scrolling=True)
        else:
            st.warning(
                "Supertree not installed. Showing tree structure below. "
                "Install with: `pip install -r requirements-visualization.txt`"
            )
            render_tree_view(tree_data)

    with tab2:
        st.json(tree_data)


# =============================================================================
# Tool Gallery (Cockpit Home)
# =============================================================================


@dataclass
class ToolCard:
    """Configuration for a tool card in the gallery."""

    id: str
    title: str
    icon: str
    description: str
    action: str  # What happens when clicked
    badge: str | None = None  # Optional badge text (e.g., count)
    badge_color: str = "#6c757d"  # Badge background color


# Default tool cards for the cockpit
DEFAULT_TOOL_CARDS = [
    ToolCard(
        id="review_queue",
        title="Review Queue",
        icon="📋",
        description="Rules needing human review, prioritized by urgency",
        action="open_review_queue",
    ),
    ToolCard(
        id="verify_all",
        title="Verify All Rules",
        icon="🔍",
        description="Run Tier 0-1 verification on all loaded rules",
        action="verify_all",
    ),
    ToolCard(
        id="error_patterns",
        title="Error Patterns",
        icon="📊",
        description="Systematic issues across the rulebook",
        action="open_error_patterns",
    ),
    ToolCard(
        id="drift_report",
        title="Drift Report",
        icon="📈",
        description="Quality changes since last baseline",
        action="open_drift_report",
    ),
    ToolCard(
        id="corpus_coverage",
        title="Corpus Coverage",
        icon="📚",
        description="Legal provisions with/without rules",
        action="open_corpus_coverage",
    ),
    ToolCard(
        id="rulebook_outline",
        title="Rulebook Outline",
        icon="🗂️",
        description="Hierarchical view of all rules by document",
        action="open_rulebook_outline",
    ),
]


def render_tool_card(card: ToolCard) -> bool:
    """Render a single tool card and return True if clicked.

    Args:
        card: ToolCard configuration

    Returns:
        True if the card's action button was clicked
    """
    with st.container():
        st.markdown(
            f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 8px;
                background: white;
            ">
                <div style="font-size: 2em; margin-bottom: 8px;">{card.icon}</div>
                <div style="font-weight: bold; margin-bottom: 4px;">{card.title}</div>
                <div style="font-size: 0.9em; color: #666;">{card.description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return st.button(
            f"Open {card.title}",
            key=f"tool_{card.id}",
            use_container_width=True,
        )


def render_tool_gallery(
    cards: list[ToolCard] | None = None,
    columns: int = 3,
) -> str | None:
    """Render the tool gallery and return action if a card is clicked.

    Args:
        cards: List of ToolCard configs (uses defaults if None)
        columns: Number of columns in the grid

    Returns:
        Action string if a card was clicked, None otherwise
    """
    if cards is None:
        cards = DEFAULT_TOOL_CARDS

    st.markdown("### Quick Actions")
    st.caption("Select a tool to get started")

    clicked_action = None

    # Create grid of cards
    cols = st.columns(columns)
    for i, card in enumerate(cards):
        with cols[i % columns]:
            if render_tool_card(card):
                clicked_action = card.action

    return clicked_action


# =============================================================================
# Insights Drawer Components
# =============================================================================


def render_insights_summary(
    total_rules: int,
    verified: int,
    needs_review: int,
    inconsistent: int,
) -> None:
    """Render a quick insights summary panel.

    Args:
        total_rules: Total number of rules
        verified: Count of verified rules
        needs_review: Count of rules needing review
        inconsistent: Count of inconsistent rules
    """
    st.markdown("### Quick Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rules", total_rules)
        st.metric("Verified", verified)
    with col2:
        st.metric("Needs Review", needs_review)
        st.metric("Inconsistent", inconsistent)

    # Progress toward full verification
    if total_rules > 0:
        verified_pct = verified / total_rules
        st.progress(verified_pct, text=f"{verified_pct:.0%} verified")


def render_recent_activity(
    recent_verifications: list[tuple[str, str, str]],
    max_items: int = 5,
) -> None:
    """Render recent verification activity.

    Args:
        recent_verifications: List of (rule_id, status, timestamp) tuples
        max_items: Maximum items to show
    """
    from frontend.ui.review_helpers import get_status_color, get_status_emoji

    st.markdown("### Recent Activity")

    if not recent_verifications:
        st.info("No recent verifications")
        return

    for rule_id, status, timestamp in recent_verifications[:max_items]:
        color = get_status_color(status)
        emoji = get_status_emoji(status)
        st.markdown(
            f"<div style='padding:4px 0;'>"
            f"<span style='color:{color};'>{emoji}</span> "
            f"<strong>{rule_id}</strong> "
            f"<span style='color:#666;font-size:0.85em;'>{timestamp}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
