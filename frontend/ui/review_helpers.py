"""
Review helpers - Shared review/verification UI logic.

This module contains reusable functions for:
- Status color/emoji mapping
- Priority scoring for review queue
- Human review submission
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.rule_service.app.services.schema import Rule, ConsistencyBlock


# =============================================================================
# Status Display Helpers
# =============================================================================


def get_status_color(status: str) -> str:
    """Map consistency status to display color.

    Args:
        status: One of verified, needs_review, inconsistent, unverified

    Returns:
        Hex color code for the status
    """
    return {
        "verified": "#28a745",      # green
        "needs_review": "#ffc107",  # yellow/amber
        "inconsistent": "#dc3545",  # red
        "unverified": "#6c757d",    # gray
    }.get(status, "#6c757d")


def get_status_emoji(status: str) -> str:
    """Map consistency status to emoji indicator.

    Args:
        status: One of verified, needs_review, inconsistent, unverified

    Returns:
        Emoji character for the status
    """
    return {
        "verified": "✓",
        "needs_review": "?",
        "inconsistent": "✗",
        "unverified": "○",
    }.get(status, "○")


def get_priority_emoji(priority: float) -> str:
    """Get emoji indicator for priority level.

    Args:
        priority: Priority score from 0.0 to 1.0

    Returns:
        Colored emoji indicator
    """
    if priority > 0.8:
        return "🔴"  # High priority
    elif priority > 0.5:
        return "🟡"  # Medium priority
    else:
        return "🟢"  # Low priority


# =============================================================================
# Priority Scoring
# =============================================================================


def get_priority_score(
    rule: Rule,
    verification_results: dict[str, ConsistencyBlock],
) -> float:
    """Calculate review priority score (higher = more urgent).

    Priority is based on:
    - Consistency status (inconsistent > needs_review > unverified > verified)
    - Number of failures and warnings

    Args:
        rule: The rule to score
        verification_results: Dict mapping rule_id to ConsistencyBlock

    Returns:
        Priority score from 0.0 to 1.0
    """
    from backend.rule_service.app.services.schema import ConsistencyStatus

    if rule.rule_id not in verification_results:
        return 0.5  # Unverified rules have medium priority

    result = verification_results[rule.rule_id]

    # Base score from status
    status_score = {
        ConsistencyStatus.INCONSISTENT: 1.0,
        ConsistencyStatus.NEEDS_REVIEW: 0.7,
        ConsistencyStatus.UNVERIFIED: 0.5,
        ConsistencyStatus.VERIFIED: 0.1,
    }.get(result.summary.status, 0.5)

    # Adjust by failure/warning count
    fail_count = sum(1 for e in result.evidence if e.label == "fail")
    warn_count = sum(1 for e in result.evidence if e.label == "warning")

    priority = status_score + (fail_count * 0.1) + (warn_count * 0.05)
    return min(priority, 1.0)


def get_rule_issues(
    rule: Rule,
    verification_results: dict[str, ConsistencyBlock],
    max_issues: int = 5,
) -> list[str]:
    """Get list of issues for a rule.

    Args:
        rule: The rule to check
        verification_results: Dict mapping rule_id to ConsistencyBlock
        max_issues: Maximum number of issues to return

    Returns:
        List of issue strings (prefixed with FAIL: or WARN:)
    """
    if rule.rule_id not in verification_results:
        return ["Not yet verified"]

    result = verification_results[rule.rule_id]
    issues = []

    for ev in result.evidence:
        if ev.label == "fail":
            issues.append(f"FAIL: {ev.category}")
        elif ev.label == "warning":
            issues.append(f"WARN: {ev.category}")

    return issues[:max_issues]


# =============================================================================
# Review Submission
# =============================================================================


def submit_review(
    rule_id: str,
    label: str,
    notes: str,
    reviewer_id: str,
    rule_loader,
    verification_results: dict[str, ConsistencyBlock],
) -> bool:
    """Submit human review for a rule.

    Creates a Tier 4 human_review evidence record and updates
    the rule's consistency block.

    Args:
        rule_id: ID of the rule to review
        label: Review decision (consistent, inconsistent, unknown)
        notes: Reviewer notes
        reviewer_id: ID of the reviewer
        rule_loader: RuleLoader instance to get rule
        verification_results: Dict to update with new results

    Returns:
        True if review was submitted successfully
    """
    from backend.rule_service.app.services.schema import (
        ConsistencyBlock,
        ConsistencySummary,
        ConsistencyEvidence,
        ConsistencyStatus,
    )

    rule = rule_loader.get_rule(rule_id)
    if not rule:
        return False

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
    existing_result = verification_results.get(rule_id)
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

    # Update verification results
    verification_results[rule_id] = new_block
    rule.consistency = new_block

    return True


# =============================================================================
# UI Components
# =============================================================================


def render_status_badge(status: str, size: str = "normal") -> str:
    """Generate HTML for a status badge.

    Args:
        status: Consistency status string
        size: 'small' or 'normal'

    Returns:
        HTML string for the badge
    """
    color = get_status_color(status)
    padding = "4px 8px" if size == "small" else "8px 16px"
    font_size = "0.8em" if size == "small" else "1em"

    return (
        f'<div style="background-color:{color};color:white;'
        f'padding:{padding};border-radius:4px;text-align:center;'
        f'font-size:{font_size};font-weight:bold;">'
        f'{get_status_emoji(status)} {status.upper()}</div>'
    )


def render_priority_indicator(priority: float) -> str:
    """Generate HTML for a priority indicator.

    Args:
        priority: Priority score from 0.0 to 1.0

    Returns:
        HTML string for the indicator
    """
    emoji = get_priority_emoji(priority)
    label = "High" if priority > 0.8 else "Medium" if priority > 0.5 else "Low"

    return f"{emoji} {label} ({priority:.2f})"
