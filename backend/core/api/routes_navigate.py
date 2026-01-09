"""
Navigate endpoint for cross-border compliance navigation.

Implements v4 Flow 3: Cross-Border Navigation (Sync, Multi-Jurisdiction)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.rule_service.app.services.jurisdiction.resolver import resolve_jurisdictions, get_equivalences
from backend.rule_service.app.services.jurisdiction.evaluator import evaluate_jurisdiction
from backend.rule_service.app.services.jurisdiction.conflicts import detect_conflicts
from backend.rule_service.app.services.jurisdiction.pathway import (
    synthesize_pathway,
    aggregate_obligations,
    estimate_timeline,
)
from backend.core.ontology.jurisdiction import JurisdictionCode, ApplicableJurisdiction

router = APIRouter(prefix="/navigate", tags=["navigate"])


class NavigateRequest(BaseModel):
    """Cross-border compliance navigation request (v4 spec)."""

    issuer_jurisdiction: str = Field(
        ...,
        description="Jurisdiction code where the issuer is based",
        examples=["CH", "EU", "UK"],
    )
    target_jurisdictions: list[str] = Field(
        default_factory=list,
        description="Target market jurisdiction codes",
        examples=[["EU", "UK"]],
    )
    instrument_type: str = Field(
        ...,
        description="Type of digital asset/instrument",
        examples=["stablecoin", "tokenized_bond", "crypto_asset"],
    )
    activity: str = Field(
        ...,
        description="Regulatory activity being performed",
        examples=["public_offer", "financial_promotion", "custody"],
    )
    investor_types: list[str] = Field(
        default=["professional"],
        description="Types of investors targeted",
        examples=[["retail", "professional"]],
    )
    facts: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional facts for rule evaluation",
    )


class JurisdictionRoleResponse(BaseModel):
    """Jurisdiction with role in cross-border scenario."""

    jurisdiction: str
    regime_id: str
    role: str


class NavigateResponse(BaseModel):
    """Cross-border compliance navigation result (v4 spec)."""

    status: str = Field(
        ...,
        description="Overall status: actionable, blocked, requires_review",
    )
    applicable_jurisdictions: list[JurisdictionRoleResponse]
    jurisdiction_results: list[dict]
    conflicts: list[dict]
    pathway: list[dict]
    cumulative_obligations: list[dict]
    estimated_timeline: str
    audit_trail: list[dict]


@router.post("", response_model=NavigateResponse)
async def navigate(request: NavigateRequest) -> NavigateResponse:
    """
    Navigate cross-border compliance requirements.

    Implements v4 Flow 3: Cross-Border Navigation (Sync, Multi-Jurisdiction)

    1. resolveJurisdictions([issuer, targets])
    2. Get equivalence determinations
    3. parallelEvaluate([{jurisdiction, facts}, ...])
    4. detectConflicts(results)
    5. synthesizePathway(results, conflicts)
    6. Return {jurisdictions, conflicts, pathway, audit_trail}
    """
    audit_trail = []

    # Step 1: Resolve jurisdictions and regimes
    audit_trail.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "NAVIGATE_REQUEST",
        "details": {
            "issuer": request.issuer_jurisdiction,
            "targets": request.target_jurisdictions,
            "instrument": request.instrument_type,
            "activity": request.activity,
        },
    })

    applicable = resolve_jurisdictions(
        issuer=request.issuer_jurisdiction,
        targets=request.target_jurisdictions,
        instrument_type=request.instrument_type,
    )

    audit_trail.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "JURISDICTION_RESOLUTION",
        "details": {
            "applicable_count": len(applicable),
            "jurisdictions": [j.jurisdiction.value for j in applicable],
        },
    })

    # Step 2: Get equivalence determinations
    equivalences = get_equivalences(
        from_jurisdiction=request.issuer_jurisdiction,
        to_jurisdictions=request.target_jurisdictions,
    )

    if equivalences:
        audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "EQUIVALENCE_CHECK",
            "details": {"equivalences": equivalences},
        })

    # Step 3: Parallel evaluation across all jurisdictions
    evaluation_tasks = [
        evaluate_jurisdiction(
            jurisdiction=j.jurisdiction.value,
            regime_id=j.regime_id,
            facts={
                **request.facts,
                "instrument_type": request.instrument_type,
                "activity": request.activity,
                "investor_types": request.investor_types,
                "target_jurisdiction": j.jurisdiction.value,
            },
        )
        for j in applicable
    ]

    jurisdiction_results = await asyncio.gather(*evaluation_tasks)

    # Add role to results
    for i, result in enumerate(jurisdiction_results):
        result["role"] = applicable[i].role.value

    for result in jurisdiction_results:
        audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": f"EVALUATE_{result['jurisdiction']}",
            "details": {
                "regime": result["regime_id"],
                "rules_evaluated": result["rules_evaluated"],
                "status": result["status"],
            },
        })

    # Step 4: Detect conflicts between jurisdictions
    conflicts = detect_conflicts(jurisdiction_results)

    if conflicts:
        audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "CONFLICT_DETECTION",
            "details": {"conflicts_found": len(conflicts)},
        })

    # Step 5: Synthesize compliance pathway
    pathway = synthesize_pathway(
        results=jurisdiction_results,
        conflicts=conflicts,
        equivalences=equivalences,
    )

    # Step 6: Aggregate obligations across jurisdictions
    cumulative_obligations = aggregate_obligations(jurisdiction_results)

    # Determine overall status
    if any(c.get("severity") == "blocking" for c in conflicts):
        status = "blocked"
    elif any(c.get("severity") == "warning" for c in conflicts):
        status = "requires_review"
    elif any(jr.get("status") == "blocked" for jr in jurisdiction_results):
        status = "blocked"
    else:
        status = "actionable"

    return NavigateResponse(
        status=status,
        applicable_jurisdictions=[
            JurisdictionRoleResponse(
                jurisdiction=j.jurisdiction.value,
                regime_id=j.regime_id,
                role=j.role.value,
            )
            for j in applicable
        ],
        jurisdiction_results=jurisdiction_results,
        conflicts=conflicts,
        pathway=pathway,
        cumulative_obligations=cumulative_obligations,
        estimated_timeline=estimate_timeline(pathway),
        audit_trail=audit_trail,
    )


@router.get("/jurisdictions")
async def list_jurisdictions() -> dict:
    """List all supported jurisdictions."""
    from backend.database_service.app.services.database import get_db

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT code, name, authority FROM jurisdictions"
        )
        jurisdictions = [
            {"code": row["code"], "name": row["name"], "authority": row["authority"]}
            for row in cursor.fetchall()
        ]

    return {"jurisdictions": jurisdictions}


@router.get("/regimes")
async def list_regimes() -> dict:
    """List all regulatory regimes."""
    from backend.database_service.app.services.database import get_db

    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT id, jurisdiction_code, name, effective_date
            FROM regulatory_regimes
            ORDER BY jurisdiction_code, effective_date DESC
            """
        )
        regimes = [
            {
                "id": row["id"],
                "jurisdiction_code": row["jurisdiction_code"],
                "name": row["name"],
                "effective_date": row["effective_date"],
            }
            for row in cursor.fetchall()
        ]

    return {"regimes": regimes}


@router.get("/equivalences")
async def list_equivalences() -> dict:
    """List all equivalence determinations."""
    from backend.database_service.app.services.database import get_db

    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT id, from_jurisdiction, to_jurisdiction, scope, status, notes
            FROM equivalence_determinations
            """
        )
        equivalences = [
            {
                "id": row["id"],
                "from_jurisdiction": row["from_jurisdiction"],
                "to_jurisdiction": row["to_jurisdiction"],
                "scope": row["scope"],
                "status": row["status"],
                "notes": row["notes"],
            }
            for row in cursor.fetchall()
        ]

    return {"equivalences": equivalences}
