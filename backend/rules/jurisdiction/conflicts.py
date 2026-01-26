"""
Conflict detection for cross-jurisdiction compliance.

Detects and classifies conflicts between jurisdiction evaluation results.
"""

from __future__ import annotations

from typing import Any

from backend.core.ontology.jurisdiction import ConflictType, ConflictSeverity


# Known mutually exclusive obligation pairs
EXCLUSIVE_OBLIGATION_PAIRS: dict[frozenset[str], dict[str, str]] = {
    frozenset(["implement_cooling_off", "immediate_execution"]): {
        "description": "UK requires 24h cooling off vs immediate execution allowed elsewhere",
        "resolution": "Apply cooling off for UK-targeted offers",
    },
    frozenset(["submit_whitepaper", "no_disclosure"]): {
        "description": "Whitepaper requirement conflicts with minimal disclosure regime",
        "resolution": "Prepare whitepaper to satisfy stricter requirement",
    },
    frozenset(["add_risk_warning", "no_warning_required"]): {
        "description": "Risk warning requirement conflicts with no-warning jurisdiction",
        "resolution": "Add risk warning to satisfy stricter requirement",
    },
}


def detect_conflicts(jurisdiction_results: list[dict]) -> list[dict]:
    """
    Detect conflicts between jurisdiction evaluation results.

    Checks for:
    - Classification divergence: Same instrument, different regulatory treatment
    - Obligation conflicts: Incompatible requirements
    - Timeline conflicts: Conflicting deadlines
    - Decision conflicts: Permitted in one jurisdiction, prohibited in another

    Args:
        jurisdiction_results: List of evaluation results from each jurisdiction

    Returns:
        List of conflict dicts
    """
    conflicts = []

    # Compare each pair of jurisdictions
    for i, result_a in enumerate(jurisdiction_results):
        for result_b in jurisdiction_results[i + 1:]:
            # Skip if either has no applicable rules
            if result_a.get("status") == "no_applicable_rules":
                continue
            if result_b.get("status") == "no_applicable_rules":
                continue

            # Check decision conflicts
            decision_conflict = _check_decision_conflict(result_a, result_b)
            if decision_conflict:
                conflicts.append(decision_conflict)

            # Check obligation conflicts
            obligation_conflicts = _check_obligation_conflicts(result_a, result_b)
            conflicts.extend(obligation_conflicts)

            # Check classification divergence
            classification_conflict = _check_classification_divergence(result_a, result_b)
            if classification_conflict:
                conflicts.append(classification_conflict)

    return conflicts


def _check_decision_conflict(result_a: dict, result_b: dict) -> dict | None:
    """Check if jurisdictions have conflicting overall decisions."""
    status_a = result_a.get("status")
    status_b = result_b.get("status")

    # Compliant vs Blocked is a conflict
    if (status_a == "compliant" and status_b == "blocked") or \
       (status_a == "blocked" and status_b == "compliant"):
        blocker = result_a if status_a == "blocked" else result_b
        permitter = result_a if status_a == "compliant" else result_b

        return {
            "type": ConflictType.DECISION.value,
            "severity": ConflictSeverity.WARNING.value,
            "jurisdictions": [result_a["jurisdiction"], result_b["jurisdiction"]],
            "description": (
                f"{permitter['jurisdiction']} permits activity while "
                f"{blocker['jurisdiction']} blocks it"
            ),
            "resolution_strategy": "stricter",
            "resolution_note": (
                f"Must satisfy {blocker['jurisdiction']} requirements to proceed"
            ),
        }

    return None


def _check_obligation_conflicts(result_a: dict, result_b: dict) -> list[dict]:
    """Check for conflicting obligations between jurisdictions."""
    conflicts = []

    obls_a = {o["id"]: o for o in result_a.get("obligations", [])}
    obls_b = {o["id"]: o for o in result_b.get("obligations", [])}

    # Check for known mutually exclusive pairs
    for obl_a_id in obls_a:
        for obl_b_id in obls_b:
            pair_key = frozenset([obl_a_id, obl_b_id])
            if pair_key in EXCLUSIVE_OBLIGATION_PAIRS:
                pair_info = EXCLUSIVE_OBLIGATION_PAIRS[pair_key]
                conflicts.append({
                    "type": ConflictType.OBLIGATION.value,
                    "severity": ConflictSeverity.WARNING.value,
                    "jurisdictions": [result_a["jurisdiction"], result_b["jurisdiction"]],
                    "obligations": [obl_a_id, obl_b_id],
                    "description": pair_info["description"],
                    "resolution": pair_info["resolution"],
                    "resolution_strategy": "cumulative",
                })

    return conflicts


def _check_classification_divergence(result_a: dict, result_b: dict) -> dict | None:
    """Check if same instrument is classified differently across jurisdictions."""

    def get_classification(result: dict) -> str | None:
        for dec in result.get("decisions", []):
            if "classification" in dec.get("rule_id", "").lower():
                return dec.get("decision")
        return None

    class_a = get_classification(result_a)
    class_b = get_classification(result_b)

    if class_a and class_b and class_a != class_b:
        return {
            "type": ConflictType.CLASSIFICATION.value,
            "severity": ConflictSeverity.INFO.value,
            "jurisdictions": [result_a["jurisdiction"], result_b["jurisdiction"]],
            "description": (
                f"Classified as '{class_a}' in {result_a['jurisdiction']} "
                f"but '{class_b}' in {result_b['jurisdiction']}"
            ),
            "resolution_strategy": "cumulative",
            "resolution_note": "Must satisfy requirements for both classifications",
        }

    return None


def check_timeline_conflicts(obligations: list[dict]) -> list[dict]:
    """Check for conflicting timelines in obligations.

    Args:
        obligations: List of obligation dicts with deadlines

    Returns:
        List of timeline conflict dicts
    """
    conflicts = []

    # Group obligations by type
    by_type: dict[str, list[dict]] = {}
    for obl in obligations:
        obl_id = obl.get("id", "")
        base_type = obl_id.split("_")[0] if "_" in obl_id else obl_id
        if base_type not in by_type:
            by_type[base_type] = []
        by_type[base_type].append(obl)

    # Check for conflicting deadlines within same type
    for obl_type, obls in by_type.items():
        if len(obls) > 1:
            deadlines = [o.get("deadline") for o in obls if o.get("deadline")]
            if len(set(deadlines)) > 1:
                conflicts.append({
                    "type": ConflictType.TIMELINE.value,
                    "severity": ConflictSeverity.INFO.value,
                    "obligations": [o["id"] for o in obls],
                    "jurisdictions": list(set(o.get("jurisdiction", "") for o in obls)),
                    "description": f"Different deadlines for {obl_type} obligations",
                    "resolution_strategy": "stricter",
                    "resolution_note": "Use earliest deadline",
                })

    return conflicts
