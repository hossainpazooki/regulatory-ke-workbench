"""Jurisdiction service layer - cross-border compliance resolution and evaluation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from sqlalchemy import text

from backend.core.ontology.jurisdiction import (
    JurisdictionCode,
    ApplicableJurisdiction,
    JurisdictionRole,
    EquivalenceDetermination,
    EquivalenceStatus,
    ConflictType,
    ConflictSeverity,
)
from backend.storage.database import get_db


# =============================================================================
# Resolver Constants
# =============================================================================


DEFAULT_REGIMES: dict[str, str] = {
    "EU": "mica_2023",
    "UK": "fca_crypto_2024",
    "US": "genius_act_2025",
    "US_SEC": "securities_act_1933",
    "US_CFTC": "cftc_digital_assets_2024",
    "CH": "finsa_dlt_2021",
    "SG": "psa_2019",
    "HK": "sfc_vasp_2023",
    "JP": "psa_japan_2023",
}


# =============================================================================
# Conflict Constants
# =============================================================================


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


# =============================================================================
# Pathway Constants
# =============================================================================


STEP_TIMELINES: dict[str, dict[str, Any]] = {
    "obtain_authorization": {
        "min_days": 90,
        "max_days": 180,
        "description": "Full authorization process",
    },
    "obtain_fca_authorization": {
        "min_days": 90,
        "max_days": 180,
        "description": "FCA authorization",
    },
    "submit_whitepaper": {
        "min_days": 1,
        "max_days": 20,
        "description": "Whitepaper submission and review",
    },
    "add_risk_warning": {
        "min_days": 1,
        "max_days": 5,
        "description": "Add required disclosures",
    },
    "implement_cooling_off": {
        "min_days": 5,
        "max_days": 30,
        "description": "Technical implementation",
    },
    "conduct_assessment": {
        "min_days": 1,
        "max_days": 10,
        "description": "Appropriateness assessment implementation",
    },
    "mas_license": {
        "min_days": 60,
        "max_days": 120,
        "description": "MAS licensing process",
    },
    "finma_authorization": {
        "min_days": 90,
        "max_days": 180,
        "description": "FINMA authorization",
    },
    "maintain_approval_records": {
        "min_days": 1,
        "max_days": 5,
        "description": "Record-keeping setup",
    },
    "mlr_registration_check": {
        "min_days": 1,
        "max_days": 5,
        "description": "MLR registration verification",
    },
}


STEP_DEPENDENCIES: dict[str, list[str]] = {
    "submit_whitepaper": ["obtain_authorization"],
    "eu_passporting": ["obtain_authorization", "submit_whitepaper"],
    "uk_promotion": ["obtain_fca_authorization"],
    "add_risk_warning": [],
    "implement_cooling_off": ["add_risk_warning"],
    "conduct_assessment": ["implement_cooling_off"],
}


# =============================================================================
# Resolver Functions
# =============================================================================


def resolve_jurisdictions(
    issuer: str,
    targets: list[str],
    instrument_type: str | None = None,
) -> list[ApplicableJurisdiction]:
    """Resolve applicable jurisdictions for a cross-border scenario."""
    applicable = []

    issuer_code = issuer if isinstance(issuer, str) else issuer.value
    applicable.append(
        ApplicableJurisdiction(
            jurisdiction=JurisdictionCode(issuer_code),
            regime_id=_get_regime_for_jurisdiction(issuer_code, instrument_type),
            role=JurisdictionRole.ISSUER_HOME,
        )
    )

    for target in targets:
        target_code = target if isinstance(target, str) else target.value
        if target_code != issuer_code:
            applicable.append(
                ApplicableJurisdiction(
                    jurisdiction=JurisdictionCode(target_code),
                    regime_id=_get_regime_for_jurisdiction(target_code, instrument_type),
                    role=JurisdictionRole.TARGET,
                )
            )

    return applicable


def _get_regime_for_jurisdiction(
    jurisdiction_code: str,
    instrument_type: str | None = None,
) -> str:
    """Get the default regulatory regime for a jurisdiction."""
    return DEFAULT_REGIMES.get(jurisdiction_code, "unknown")


def get_equivalences(
    from_jurisdiction: str,
    to_jurisdictions: list[str],
) -> list[dict]:
    """Get equivalence determinations between jurisdictions."""
    if not to_jurisdictions:
        return []

    equivalences = []

    try:
        with get_db() as conn:
            target_params = {f"target_{i}": t for i, t in enumerate(to_jurisdictions)}
            placeholders = ", ".join(f":target_{i}" for i in range(len(to_jurisdictions)))

            result = conn.execute(
                text(f"""
                SELECT id, from_jurisdiction, to_jurisdiction, scope, status,
                       effective_date, expiry_date, source_reference, notes
                FROM equivalence_determinations
                WHERE from_jurisdiction = :from_j
                  AND to_jurisdiction IN ({placeholders})
                """),
                {"from_j": from_jurisdiction, **target_params},
            )

            for row in result.fetchall():
                equivalences.append({
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "scope": row[3],
                    "status": row[4],
                    "effective_date": row[5],
                    "expiry_date": row[6],
                    "source_reference": row[7],
                    "notes": row[8],
                })

            result = conn.execute(
                text(f"""
                SELECT id, from_jurisdiction, to_jurisdiction, scope, status,
                       effective_date, expiry_date, source_reference, notes
                FROM equivalence_determinations
                WHERE to_jurisdiction = :from_j
                  AND from_jurisdiction IN ({placeholders})
                """),
                {"from_j": from_jurisdiction, **target_params},
            )

            for row in result.fetchall():
                equivalences.append({
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "scope": row[3],
                    "status": row[4],
                    "effective_date": row[5],
                    "expiry_date": row[6],
                    "source_reference": row[7],
                    "notes": row[8],
                })
    except Exception:
        pass

    return equivalences


def get_jurisdiction_info(code: str) -> dict | None:
    """Get jurisdiction information from database."""
    try:
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT code, name, authority, parent_code
                FROM jurisdictions
                WHERE code = :code
                """),
                {"code": code}
            )
            row = result.fetchone()
            if row:
                return {
                    "code": row[0],
                    "name": row[1],
                    "authority": row[2],
                    "parent_code": row[3],
                }
    except Exception:
        pass
    return None


def get_regime_info(regime_id: str) -> dict | None:
    """Get regulatory regime information from database."""
    try:
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT id, jurisdiction_code, name, effective_date, sunset_date, source_url
                FROM regulatory_regimes
                WHERE id = :regime_id
                """),
                {"regime_id": regime_id}
            )
            row = result.fetchone()
            if row:
                return {
                    "id": row[0],
                    "jurisdiction_code": row[1],
                    "name": row[2],
                    "effective_date": row[3],
                    "sunset_date": row[4],
                    "source_url": row[5],
                }
    except Exception:
        pass
    return None


# =============================================================================
# Conflict Detection Functions
# =============================================================================


def detect_conflicts(jurisdiction_results: list[dict]) -> list[dict]:
    """Detect conflicts between jurisdiction evaluation results."""
    conflicts = []

    for i, result_a in enumerate(jurisdiction_results):
        for result_b in jurisdiction_results[i + 1:]:
            if result_a.get("status") == "no_applicable_rules":
                continue
            if result_b.get("status") == "no_applicable_rules":
                continue

            decision_conflict = _check_decision_conflict(result_a, result_b)
            if decision_conflict:
                conflicts.append(decision_conflict)

            obligation_conflicts = _check_obligation_conflicts(result_a, result_b)
            conflicts.extend(obligation_conflicts)

            classification_conflict = _check_classification_divergence(result_a, result_b)
            if classification_conflict:
                conflicts.append(classification_conflict)

    return conflicts


def _check_decision_conflict(result_a: dict, result_b: dict) -> dict | None:
    """Check if jurisdictions have conflicting overall decisions."""
    status_a = result_a.get("status")
    status_b = result_b.get("status")

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
    """Check for conflicting timelines in obligations."""
    conflicts = []

    by_type: dict[str, list[dict]] = {}
    for obl in obligations:
        obl_id = obl.get("id", "")
        base_type = obl_id.split("_")[0] if "_" in obl_id else obl_id
        if base_type not in by_type:
            by_type[base_type] = []
        by_type[base_type].append(obl)

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


# =============================================================================
# Pathway Synthesis Functions
# =============================================================================


def synthesize_pathway(
    results: list[dict],
    conflicts: list[dict],
    equivalences: list[dict],
) -> list[dict]:
    """Synthesize ordered compliance pathway from evaluation results."""
    steps = []
    step_id = 1

    sorted_results = sorted(
        results,
        key=lambda r: (0 if "issuer" in r.get("role", "") else 1, r["jurisdiction"]),
    )

    for result in sorted_results:
        jurisdiction = result["jurisdiction"]
        regime_id = result.get("regime_id", "unknown")

        for obligation in result.get("obligations", []):
            obl_id = obligation.get("id", "")

            step = {
                "step_id": step_id,
                "jurisdiction": jurisdiction,
                "regime": regime_id,
                "obligation_id": obl_id,
                "action": obligation.get("description", obl_id),
                "source": obligation.get("source_ref") or obligation.get("source"),
                "prerequisites": [],
                "timeline": STEP_TIMELINES.get(
                    obl_id,
                    {"min_days": 30, "max_days": 90, "description": "Compliance step"},
                ),
                "status": "pending",
                "waiver_reason": None,
            }

            if obl_id in STEP_DEPENDENCIES:
                for prereq_id in STEP_DEPENDENCIES[obl_id]:
                    prereq_step = next(
                        (s for s in steps if s["obligation_id"] == prereq_id),
                        None,
                    )
                    if prereq_step:
                        step["prerequisites"].append(prereq_step["step_id"])

            steps.append(step)
            step_id += 1

    for equiv in equivalences:
        if equiv.get("status") == "equivalent":
            for step in steps:
                if step["jurisdiction"] == equiv.get("to"):
                    if step["obligation_id"] in ["obtain_authorization", "submit_whitepaper"]:
                        step["status"] = "waived"
                        step["waiver_reason"] = (
                            f"Equivalent recognition from {equiv.get('from')} "
                            f"({equiv.get('scope')})"
                        )

    return steps


def aggregate_obligations(results: list[dict]) -> list[dict]:
    """Aggregate and deduplicate obligations across all jurisdictions."""
    seen: set[tuple[str, str]] = set()
    obligations = []

    for result in results:
        for obl in result.get("obligations", []):
            key = (obl.get("id", ""), result.get("jurisdiction", ""))
            if key not in seen:
                seen.add(key)
                obligations.append({
                    **obl,
                    "jurisdiction": result.get("jurisdiction"),
                    "regime": result.get("regime_id"),
                })

    return sorted(obligations, key=lambda o: (o.get("jurisdiction", ""), o.get("id", "")))


def estimate_timeline(pathway: list[dict]) -> str:
    """Calculate overall timeline estimate from pathway."""
    if not pathway:
        return "N/A"

    total_max_days = sum(
        step.get("timeline", {}).get("max_days", 30)
        for step in pathway
        if step.get("status") != "waived"
    )

    estimated_days = int(total_max_days * 0.6)

    if estimated_days < 30:
        return "< 1 month"
    elif estimated_days < 90:
        return "1-3 months"
    elif estimated_days < 180:
        return "3-6 months"
    else:
        return "6-12 months"


def get_critical_path(pathway: list[dict]) -> list[dict]:
    """Identify the critical path through the compliance pathway."""
    if not pathway:
        return []

    step_by_id = {s["step_id"]: s for s in pathway}
    path_lengths: dict[int, int] = {}

    def get_path_length(step_id: int) -> int:
        if step_id in path_lengths:
            return path_lengths[step_id]

        step = step_by_id.get(step_id)
        if not step:
            return 0

        prereqs = step.get("prerequisites", [])
        if not prereqs:
            length = step.get("timeline", {}).get("max_days", 30)
        else:
            max_prereq = max(get_path_length(p) for p in prereqs)
            length = max_prereq + step.get("timeline", {}).get("max_days", 30)

        path_lengths[step_id] = length
        return length

    for step in pathway:
        get_path_length(step["step_id"])

    if not path_lengths:
        return []

    critical_end = max(path_lengths, key=path_lengths.get)

    critical_path = []
    current = critical_end

    while current:
        step = step_by_id.get(current)
        if not step:
            break
        critical_path.append(step)
        prereqs = step.get("prerequisites", [])
        if prereqs:
            current = max(prereqs, key=lambda p: path_lengths.get(p, 0))
        else:
            current = None

    return list(reversed(critical_path))


# =============================================================================
# Evaluator Functions
# =============================================================================


# Global rule loader singleton
_rule_loader = None


def _get_rule_loader():
    """Get or create the global rule loader."""
    global _rule_loader
    if _rule_loader is None:
        from backend.rules import RuleLoader
        _rule_loader = RuleLoader()
        try:
            _rule_loader.load_directory("backend/rules/data")
        except FileNotFoundError:
            pass
    return _rule_loader


async def evaluate_jurisdiction(
    jurisdiction: str | JurisdictionCode,
    regime_id: str,
    facts: dict[str, Any],
) -> dict:
    """Evaluate facts against all applicable rules in a jurisdiction."""
    from backend.core.ontology.scenario import Scenario
    from backend.rules import DecisionEngine

    if isinstance(jurisdiction, JurisdictionCode):
        jurisdiction_code = jurisdiction.value
    else:
        jurisdiction_code = jurisdiction

    loader = _get_rule_loader()
    engine = DecisionEngine(loader=loader)

    # Get rules for this jurisdiction
    all_rules = loader.get_all_rules()
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

    # Create scenario from facts
    safe_facts = {}
    for key, value in facts.items():
        if isinstance(value, (list, dict)):
            continue
        safe_facts[key] = value

    scenario = Scenario(**safe_facts)

    # Evaluate each applicable rule
    decisions = []
    all_obligations = []
    rules_evaluated = 0

    for rule in jurisdiction_rules:
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

            for obl in result.obligations:
                all_obligations.append({
                    "id": obl.id,
                    "description": obl.description,
                    "deadline": obl.deadline,
                    "rule_id": rule.rule_id,
                    "jurisdiction": jurisdiction_code,
                })

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
    jurisdictions: list[tuple[str, str]],
    facts: dict[str, Any],
) -> list[dict]:
    """Evaluate facts across multiple jurisdictions in parallel."""
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
    """Synchronous wrapper for evaluate_jurisdiction."""
    return asyncio.run(evaluate_jurisdiction(jurisdiction, regime_id, facts))
