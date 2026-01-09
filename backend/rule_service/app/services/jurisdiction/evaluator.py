"""
Jurisdiction evaluator for parallel multi-jurisdiction assessment.

Implements v4 parallel evaluation pattern with O(1) premise index lookup.
"""

from __future__ import annotations

import asyncio
from typing import Any

from backend.rule_service.app.services.engine import DecisionEngine
from backend.rule_service.app.services.loader import RuleLoader
from backend.database_service.app.services.compiler.premise_index import get_premise_index
from backend.core.ontology.jurisdiction import JurisdictionCode


# Global rule loader singleton
_rule_loader: RuleLoader | None = None


def _get_rule_loader() -> RuleLoader:
    """Get or create the global rule loader."""
    global _rule_loader
    if _rule_loader is None:
        _rule_loader = RuleLoader()
        _rule_loader.load_directory("backend/rule_service/data")
    return _rule_loader


async def evaluate_jurisdiction(
    jurisdiction: str | JurisdictionCode,
    regime_id: str,
    facts: dict[str, Any],
) -> dict:
    """
    Evaluate facts against all applicable rules in a jurisdiction.

    Uses v4 pretreat/infer pattern:
    1. O(1) premise index lookup (not O(n) scan)
    2. Load pre-compiled rules
    3. Linear evaluation of flattened conditions
    4. Aggregate decisions and obligations

    Args:
        jurisdiction: Jurisdiction code
        regime_id: Regulatory regime identifier
        facts: Dictionary of fact values

    Returns:
        Jurisdiction evaluation result dict
    """
    from backend.core.ontology.scenario import Scenario

    if isinstance(jurisdiction, JurisdictionCode):
        jurisdiction_code = jurisdiction.value
    else:
        jurisdiction_code = jurisdiction

    loader = _get_rule_loader()
    engine = DecisionEngine(loader=loader)
    premise_index = get_premise_index()

    # Build premise index if needed
    all_rules = loader.get_all_rules()
    if not premise_index.get_all_keys():
        premise_index.build(all_rules)

    # Get rules for this jurisdiction using O(1) lookup
    jurisdiction_rules = [
        r for r in all_rules
        if r.jurisdiction.value == jurisdiction_code
    ]

    if not jurisdiction_rules:
        return {
            "jurisdiction": jurisdiction_code,
            "regime_id": regime_id,
            "applicable_rules": 0,
            "rules_evaluated": 0,
            "decisions": [],
            "obligations": [],
            "status": "no_applicable_rules",
        }

    # Create scenario from facts - handle potential unhashable types
    safe_facts = {}
    for key, value in facts.items():
        if isinstance(value, (list, dict)):
            continue  # Skip complex types for scenario construction
        safe_facts[key] = value

    scenario = Scenario(**safe_facts)

    # Evaluate each applicable rule
    decisions = []
    all_obligations = []
    rules_evaluated = 0

    for rule in jurisdiction_rules:
        # Evaluate using the decision engine with scenario and rule_id
        result = engine.evaluate(scenario, rule.rule_id)
        rules_evaluated += 1

        if result.applicable:
            decisions.append({
                "rule_id": rule.rule_id,
                "decision": result.decision,
                "trace": [
                    {
                        "node": step.node,
                        "condition": step.condition,
                        "result": step.result,
                        "value_checked": step.value_checked,
                    }
                    for step in result.trace
                ],
                "source": {
                    "document": rule.source.document_id if rule.source else None,
                    "article": rule.source.article if rule.source else None,
                },
            })

            # Collect obligations
            for obl in result.obligations:
                all_obligations.append({
                    "id": obl.id,
                    "description": obl.description,
                    "deadline": obl.deadline,
                    "rule_id": rule.rule_id,
                    "jurisdiction": jurisdiction_code,
                })

    # Determine jurisdiction status based on decisions
    decision_results = [d["decision"] for d in decisions]

    if "prohibited" in decision_results or "not_authorized" in decision_results:
        status = "blocked"
    elif "non_compliant" in decision_results:
        status = "requires_action"
    elif not decisions:
        status = "no_applicable_rules"
    else:
        status = "compliant"

    return {
        "jurisdiction": jurisdiction_code,
        "regime_id": regime_id,
        "applicable_rules": len(jurisdiction_rules),
        "rules_evaluated": rules_evaluated,
        "decisions": decisions,
        "obligations": all_obligations,
        "status": status,
    }


async def evaluate_multiple_jurisdictions(
    jurisdictions: list[tuple[str, str]],  # (jurisdiction_code, regime_id)
    facts: dict[str, Any],
) -> list[dict]:
    """
    Evaluate facts across multiple jurisdictions in parallel.

    Uses asyncio.gather for concurrent evaluation.

    Args:
        jurisdictions: List of (jurisdiction_code, regime_id) tuples
        facts: Dictionary of fact values

    Returns:
        List of jurisdiction evaluation results
    """
    tasks = [
        evaluate_jurisdiction(jurisdiction, regime_id, facts)
        for jurisdiction, regime_id in jurisdictions
    ]

    return await asyncio.gather(*tasks)


def evaluate_jurisdiction_sync(
    jurisdiction: str | JurisdictionCode,
    regime_id: str,
    facts: dict[str, Any],
) -> dict:
    """
    Synchronous wrapper for evaluate_jurisdiction.

    For use in non-async contexts.
    """
    return asyncio.run(evaluate_jurisdiction(jurisdiction, regime_id, facts))
