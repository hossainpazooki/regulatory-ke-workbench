"""
Worklist module - Review queue and navigator components.

This module provides the worklist/queue panel for the cockpit layout:
- WorklistItem dataclass for queue entries
- Filtering and sorting utilities
- Compact worklist rendering
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import streamlit as st

if TYPE_CHECKING:
    from backend.rule_service.app.services.schema import Rule, ConsistencyBlock


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class WorklistItem:
    """A single item in the review worklist."""

    rule_id: str
    status: str  # verified, needs_review, inconsistent, unverified
    priority: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    document_id: str | None = None
    article: str | None = None
    fail_count: int = 0
    warn_count: int = 0
    last_verified: str | None = None

    @property
    def source_label(self) -> str:
        """Generate a source label like 'MiCA Art.36'."""
        if self.document_id:
            doc = self.document_id.replace("_", " ").title()
            if self.article:
                return f"{doc} Art.{self.article}"
            return doc
        return ""

    @property
    def priority_emoji(self) -> str:
        """Get emoji indicator for priority."""
        if self.priority > 0.8:
            return "🔴"
        elif self.priority > 0.5:
            return "🟡"
        return "🟢"

    @property
    def status_emoji(self) -> str:
        """Get emoji indicator for status."""
        return {
            "verified": "✓",
            "needs_review": "?",
            "inconsistent": "✗",
            "unverified": "○",
        }.get(self.status, "○")


# =============================================================================
# Worklist Building
# =============================================================================


def build_worklist(
    rules: list[Rule],
    verification_results: dict[str, ConsistencyBlock],
) -> list[WorklistItem]:
    """Build a worklist from rules and verification results.

    Args:
        rules: List of all rules
        verification_results: Dict mapping rule_id to ConsistencyBlock

    Returns:
        List of WorklistItem, sorted by priority (highest first)
    """
    from frontend.ui.review_helpers import get_priority_score

    items = []

    for rule in rules:
        result = verification_results.get(rule.rule_id)

        if result:
            status = result.summary.status
            if hasattr(status, 'value'):
                status = status.value
            confidence = result.summary.confidence
            last_verified = result.summary.last_verified
            fail_count = sum(1 for e in result.evidence if e.label == "fail")
            warn_count = sum(1 for e in result.evidence if e.label == "warning")
        else:
            status = "unverified"
            confidence = 0.0
            last_verified = None
            fail_count = 0
            warn_count = 0

        priority = get_priority_score(rule, verification_results)

        item = WorklistItem(
            rule_id=rule.rule_id,
            status=status,
            priority=priority,
            confidence=confidence,
            document_id=rule.source.document_id if rule.source else None,
            article=rule.source.article if rule.source else None,
            fail_count=fail_count,
            warn_count=warn_count,
            last_verified=last_verified,
        )
        items.append(item)

    # Sort by priority (descending)
    items.sort(key=lambda x: x.priority, reverse=True)

    return items


def filter_worklist(
    items: list[WorklistItem],
    status_filter: str = "all",
    document_filter: str | None = None,
    min_priority: float = 0.0,
) -> list[WorklistItem]:
    """Filter worklist items.

    Args:
        items: List of WorklistItem to filter
        status_filter: 'all', 'needs_review', 'inconsistent', 'unverified', 'verified'
        document_filter: Filter by document_id (None for all)
        min_priority: Minimum priority score

    Returns:
        Filtered list of WorklistItem
    """
    result = items

    # Status filter
    if status_filter == "needs_review":
        result = [i for i in result if i.status in ("needs_review", "inconsistent", "unverified")]
    elif status_filter != "all":
        result = [i for i in result if i.status == status_filter]

    # Document filter
    if document_filter:
        result = [i for i in result if i.document_id == document_filter]

    # Priority filter
    if min_priority > 0:
        result = [i for i in result if i.priority >= min_priority]

    return result


# =============================================================================
# Worklist Rendering
# =============================================================================


def render_worklist_item(
    item: WorklistItem,
    selected: bool = False,
    show_issues: bool = True,
) -> bool:
    """Render a single worklist item and return True if clicked.

    Args:
        item: The WorklistItem to render
        selected: Whether this item is currently selected
        show_issues: Whether to show fail/warn counts

    Returns:
        True if the item was clicked
    """
    from frontend.ui.review_helpers import get_status_color

    status_color = get_status_color(item.status)

    # Use a container with border styling
    with st.container(border=True):
        # Top row: emoji + rule_id + badges + confidence
        col1, col2 = st.columns([3, 1])
        with col1:
            # Build badge text
            badge_text = ""
            if show_issues and item.fail_count > 0:
                badge_text += f" :red[{item.fail_count}F]"
            if show_issues and item.warn_count > 0:
                badge_text += f" :orange[{item.warn_count}W]"

            st.markdown(f"**{item.priority_emoji} {item.rule_id}**{badge_text}")

        with col2:
            st.caption(f"{item.confidence:.0%}")

        # Source label
        if item.source_label:
            st.caption(item.source_label)

        # Select button
        return st.button(
            "Select",
            key=f"wl_{item.rule_id}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        )


def render_worklist_panel(
    items: list[WorklistItem],
    selected_rule_id: str | None,
    max_items: int = 20,
) -> str | None:
    """Render the full worklist panel and return selected rule_id.

    Args:
        items: List of WorklistItem to display
        selected_rule_id: Currently selected rule ID
        max_items: Maximum items to show

    Returns:
        Rule ID if an item was clicked, None otherwise
    """
    st.markdown("### Review Queue")
    st.caption(f"{len(items)} rules")

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Status",
            options=["needs_review", "all", "inconsistent", "unverified", "verified"],
            key="wl_status_filter",
            label_visibility="collapsed",
        )
    with col2:
        # Get unique documents
        documents = sorted(set(i.document_id for i in items if i.document_id))
        doc_options = ["All documents"] + documents
        doc_filter = st.selectbox(
            "Document",
            options=doc_options,
            key="wl_doc_filter",
            label_visibility="collapsed",
        )

    # Apply filters
    filtered = filter_worklist(
        items,
        status_filter=status_filter,
        document_filter=doc_filter if doc_filter != "All documents" else None,
    )

    st.caption(f"Showing {min(len(filtered), max_items)} of {len(filtered)} filtered")

    st.divider()

    # Render items
    clicked_id = None
    for item in filtered[:max_items]:
        is_selected = item.rule_id == selected_rule_id
        if render_worklist_item(item, selected=is_selected):
            clicked_id = item.rule_id

    if len(filtered) > max_items:
        st.caption(f"...and {len(filtered) - max_items} more")

    return clicked_id


# =============================================================================
# Navigator Mode (Alternative to Queue)
# =============================================================================


def render_navigator_panel(
    rules: list,  # List[Rule]
    selected_rule_id: str | None,
    verification_results: dict,
) -> str | None:
    """Render a simple rule navigator panel.

    This is a simpler alternative to the review queue, showing
    all rules grouped by document.

    Args:
        rules: List of all rules
        selected_rule_id: Currently selected rule ID
        verification_results: Dict mapping rule_id to ConsistencyBlock

    Returns:
        Rule ID if a rule was selected, None otherwise
    """
    from frontend.ui.review_helpers import get_status_color, get_status_emoji

    st.markdown("### Rules Navigator")

    # Group by document
    by_doc: dict[str, list] = {}
    for rule in rules:
        doc_id = rule.source.document_id if rule.source else "Other"
        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(rule)

    clicked_id = None

    for doc_id, doc_rules in sorted(by_doc.items()):
        doc_title = doc_id.replace("_", " ").title()
        with st.expander(f"**{doc_title}** ({len(doc_rules)})", expanded=True):
            for rule in doc_rules:
                result = verification_results.get(rule.rule_id)
                if result:
                    status = result.summary.status
                    if hasattr(status, 'value'):
                        status = status.value
                else:
                    status = "unverified"

                color = get_status_color(status)
                emoji = get_status_emoji(status)
                is_selected = rule.rule_id == selected_rule_id

                # Simple button with status indicator
                label = f"{emoji} {rule.rule_id}"
                if rule.source and rule.source.article:
                    label += f" (Art.{rule.source.article})"

                btn_type = "primary" if is_selected else "secondary"
                if st.button(label, key=f"nav_{rule.rule_id}", use_container_width=True, type=btn_type):
                    clicked_id = rule.rule_id

    return clicked_id
