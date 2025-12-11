"""
Review Queue - Prioritized list of rules needing human review.

This page shows rules that need attention based on their consistency status,
allowing KE team members to submit Tier 4 human reviews.
"""

import sys
from pathlib import Path

# Add backend to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from backend.rules import RuleLoader, Rule
from backend.rules.schema import ConsistencyStatus
from backend.verify import ConsistencyEngine
from backend.analytics import ErrorPatternAnalyzer

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Review Queue - KE Workbench",
    page_icon="📋",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

if "rule_loader" not in st.session_state:
    st.session_state.rule_loader = RuleLoader()
    rules_dir = Path(__file__).parent.parent.parent / "backend" / "rules"
    try:
        st.session_state.rule_loader.load_directory(rules_dir)
    except FileNotFoundError:
        pass

if "consistency_engine" not in st.session_state:
    st.session_state.consistency_engine = ConsistencyEngine()

if "verification_results" not in st.session_state:
    st.session_state.verification_results = {}


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_status_color(status: str) -> str:
    """Map consistency status to display color."""
    return {
        "verified": "#28a745",
        "needs_review": "#ffc107",
        "inconsistent": "#dc3545",
        "unverified": "#6c757d",
    }.get(status, "#6c757d")


def get_priority_score(rule: Rule) -> float:
    """Calculate review priority score (higher = more urgent)."""
    if rule.rule_id not in st.session_state.verification_results:
        return 0.5  # Unverified rules have medium priority

    result = st.session_state.verification_results[rule.rule_id]

    # Base score from status
    status_score = {
        ConsistencyStatus.INCONSISTENT: 1.0,
        ConsistencyStatus.NEEDS_REVIEW: 0.7,
        ConsistencyStatus.UNVERIFIED: 0.5,
        ConsistencyStatus.VERIFIED: 0.1,
    }.get(result.summary.status, 0.5)

    # Adjust by failure count
    fail_count = sum(1 for e in result.evidence if e.label == "fail")
    warn_count = sum(1 for e in result.evidence if e.label == "warning")

    priority = status_score + (fail_count * 0.1) + (warn_count * 0.05)
    return min(priority, 1.0)


def get_rule_issues(rule: Rule) -> list[str]:
    """Get list of issues for a rule."""
    if rule.rule_id not in st.session_state.verification_results:
        return ["Not yet verified"]

    result = st.session_state.verification_results[rule.rule_id]
    issues = []

    for ev in result.evidence:
        if ev.label == "fail":
            issues.append(f"FAIL: {ev.category}")
        elif ev.label == "warning":
            issues.append(f"WARN: {ev.category}")

    return issues[:5]  # Limit to 5 issues


def submit_review(rule_id: str, label: str, notes: str, reviewer_id: str) -> bool:
    """Submit human review for a rule."""
    rule = st.session_state.rule_loader.get_rule(rule_id)
    if not rule:
        return False

    from datetime import datetime, timezone
    from backend.rules.schema import (
        ConsistencyBlock,
        ConsistencySummary,
        ConsistencyEvidence,
    )

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Map label to score
    score = {"consistent": 1.0, "inconsistent": 0.0, "unknown": 0.5}.get(label, 0.5)

    human_evidence = ConsistencyEvidence(
        tier=4,
        category="human_review",
        label="pass" if label == "consistent" else (
            "fail" if label == "inconsistent" else "warning"
        ),
        score=score,
        details=f"Human review by {reviewer_id}: {notes}",
        rule_element="__rule__",
        timestamp=timestamp,
    )

    # Get existing evidence
    existing_result = st.session_state.verification_results.get(rule_id)
    if existing_result:
        existing_evidence = list(existing_result.evidence)
        existing_evidence.append(human_evidence)
        existing_conf = existing_result.summary.confidence
        new_confidence = (0.4 * existing_conf) + (0.6 * score)
    else:
        existing_evidence = [human_evidence]
        new_confidence = score

    new_status = {
        "consistent": ConsistencyStatus.VERIFIED,
        "inconsistent": ConsistencyStatus.INCONSISTENT,
        "unknown": ConsistencyStatus.NEEDS_REVIEW,
    }.get(label, ConsistencyStatus.NEEDS_REVIEW)

    new_summary = ConsistencySummary(
        status=new_status,
        confidence=round(new_confidence, 4),
        last_verified=timestamp,
        verified_by=f"human:{reviewer_id}",
        notes=notes,
    )

    new_block = ConsistencyBlock(
        summary=new_summary,
        evidence=existing_evidence,
    )

    # Update session state
    st.session_state.verification_results[rule_id] = new_block
    rule.consistency = new_block

    return True


# -----------------------------------------------------------------------------
# Main Content
# -----------------------------------------------------------------------------

st.title("📋 Rule Review Queue")
st.caption("Prioritized list of rules needing human review")

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    status_filter = st.selectbox(
        "Filter by status",
        options=["All needing review", "inconsistent", "needs_review", "unverified"],
    )

with col2:
    tier_filter = st.selectbox(
        "Missing verification tier",
        options=["Any", "tier2", "tier4"],
    )

with col3:
    reviewer_id = st.text_input("Reviewer ID", value="reviewer_1")

st.divider()

# Get rules and build queue
all_rules = st.session_state.rule_loader.get_all_rules()

# Filter rules
queue = []
for rule in all_rules:
    needs_review = False

    # Check if already verified
    result = st.session_state.verification_results.get(rule.rule_id)

    if result is None:
        needs_review = True
    else:
        status = result.summary.status

        if status_filter == "All needing review":
            if status in (
                ConsistencyStatus.INCONSISTENT,
                ConsistencyStatus.NEEDS_REVIEW,
                ConsistencyStatus.UNVERIFIED,
            ):
                needs_review = True
        elif status_filter == "inconsistent":
            if status == ConsistencyStatus.INCONSISTENT:
                needs_review = True
        elif status_filter == "needs_review":
            if status == ConsistencyStatus.NEEDS_REVIEW:
                needs_review = True
        elif status_filter == "unverified":
            if status == ConsistencyStatus.UNVERIFIED:
                needs_review = True

        # Check for missing tier
        if tier_filter != "Any" and needs_review:
            tier_num = int(tier_filter.replace("tier", ""))
            tiers_present = {e.tier for e in result.evidence}
            if tier_num not in tiers_present:
                needs_review = True
            else:
                needs_review = False

    if needs_review:
        queue.append(rule)

# Sort by priority
queue.sort(key=get_priority_score, reverse=True)

st.write(f"**{len(queue)} rules need review**")

# Display queue
for rule in queue[:20]:  # Limit to 20
    result = st.session_state.verification_results.get(rule.rule_id)
    priority = get_priority_score(rule)

    if result:
        status = result.summary.status.value if hasattr(result.summary.status, 'value') else str(result.summary.status)
        confidence = result.summary.confidence
    else:
        status = "unverified"
        confidence = 0.0

    color = get_status_color(status)

    # Expander for each rule
    with st.expander(
        f"{'🔴' if priority > 0.8 else '🟡' if priority > 0.5 else '🟢'} {rule.rule_id}",
        expanded=False,
    ):
        # Status header
        status_col, conf_col, priority_col = st.columns([1, 1, 1])

        with status_col:
            st.markdown(
                f'<div style="background-color:{color};color:white;'
                f'padding:4px 8px;border-radius:4px;text-align:center;">'
                f'{status.upper()}</div>',
                unsafe_allow_html=True,
            )

        with conf_col:
            st.metric("Confidence", f"{confidence:.0%}")

        with priority_col:
            st.metric("Priority", f"{priority:.2f}")

        # Source reference
        if rule.source:
            st.caption(f"Source: {rule.source.document_id} Art. {rule.source.article or ''}")

        # Issues list
        issues = get_rule_issues(rule)
        if issues:
            st.markdown("**Issues:**")
            for issue in issues:
                if issue.startswith("FAIL"):
                    st.markdown(f"- :red[{issue}]")
                elif issue.startswith("WARN"):
                    st.markdown(f"- :orange[{issue}]")
                else:
                    st.markdown(f"- {issue}")

        # Existing evidence summary
        if result and result.evidence:
            with st.expander("Verification history", expanded=False):
                for ev in result.evidence[-5:]:  # Last 5
                    tier_label = f"Tier {ev.tier}"
                    st.markdown(f"**{tier_label}** ({ev.category}): {ev.label}")
                    st.caption(ev.details[:100])

        st.divider()

        # Actions row
        action_col1, action_col2 = st.columns([1, 2])

        with action_col1:
            if st.button(
                "🔄 Run verification",
                key=f"verify_{rule.rule_id}",
                use_container_width=True,
            ):
                with st.spinner("Verifying..."):
                    result = st.session_state.consistency_engine.verify_rule(rule)
                    st.session_state.verification_results[rule.rule_id] = result
                st.success("Verification complete")
                st.rerun()

        with action_col2:
            st.markdown("**Submit review:**")

            review_label = st.radio(
                "Decision",
                options=["consistent", "inconsistent", "unknown"],
                key=f"label_{rule.rule_id}",
                horizontal=True,
            )

            review_notes = st.text_area(
                "Notes",
                key=f"notes_{rule.rule_id}",
                placeholder="Explain your decision...",
                height=80,
            )

            if st.button(
                "✅ Submit review",
                key=f"submit_{rule.rule_id}",
                type="primary",
                use_container_width=True,
            ):
                if not review_notes:
                    st.error("Please provide review notes")
                elif not reviewer_id:
                    st.error("Please enter your reviewer ID")
                else:
                    success = submit_review(
                        rule.rule_id,
                        review_label,
                        review_notes,
                        reviewer_id,
                    )
                    if success:
                        st.success("Review submitted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to submit review")

# Footer
st.divider()
st.caption("Review Queue v0.1 | Internal Use Only")
