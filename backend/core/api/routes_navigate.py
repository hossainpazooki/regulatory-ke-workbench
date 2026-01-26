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

from backend.rules.jurisdiction import (
    resolve_jurisdictions,
    get_equivalences,
    evaluate_jurisdiction,
    detect_conflicts,
    synthesize_pathway,
    aggregate_obligations,
    estimate_timeline,
)
from backend.core.ontology.jurisdiction import JurisdictionCode, ApplicableJurisdiction
from backend.token_compliance import (
    analyze_token_compliance,
    TokenStandard,
)
from backend.protocol_risk import (
    assess_protocol_risk,
    get_protocol_defaults,
    PROTOCOL_DEFAULTS,
)
from backend.defi_risk import (
    score_defi_protocol,
    DEFI_PROTOCOL_DEFAULTS,
    SmartContractRisk,
    EconomicRisk,
    OracleRisk,
    GovernanceRisk,
)

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
    token_standard: str | None = Field(
        None,
        description="Token standard (ERC-20, BEP-20, SPL, etc.)",
        examples=["ERC-20", "BEP-20", "SPL"],
    )
    underlying_chain: str | None = Field(
        None,
        description="Underlying blockchain protocol",
        examples=["ethereum", "solana", "polygon", "avalanche"],
    )
    is_defi_integrated: bool = Field(
        False,
        description="Whether the instrument integrates with DeFi protocols",
    )
    defi_protocol: str | None = Field(
        None,
        description="DeFi protocol name if integrated",
        examples=["aave_v3", "uniswap_v3", "lido", "gmx"],
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
    # Market risk enhancements
    token_compliance: dict | None = Field(
        None,
        description="Token standard compliance analysis (Howey test, GENIUS Act)",
    )
    protocol_risk: dict | None = Field(
        None,
        description="Underlying blockchain protocol risk assessment",
    )
    defi_risk: dict | None = Field(
        None,
        description="DeFi protocol risk score if integrated",
    )


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

    # Market Risk Enhancements: Token compliance analysis
    token_compliance_result = None
    if request.token_standard:
        try:
            # Map string token standard to enum
            token_standard_map = {
                "erc-20": TokenStandard.ERC_20,
                "erc-721": TokenStandard.ERC_721,
                "erc-1155": TokenStandard.ERC_1155,
                "bep-20": TokenStandard.BEP_20,
                "spl": TokenStandard.SPL,
                "trc-20": TokenStandard.TRC_20,
            }
            standard_key = request.token_standard.lower()
            token_standard_enum = token_standard_map.get(standard_key, TokenStandard.ERC_20)

            # Derive compliance parameters from instrument type and facts
            is_stablecoin = request.instrument_type in ["stablecoin", "payment_stablecoin"]
            is_security_like = request.instrument_type in ["tokenized_bond", "security_token", "tokenized_equity"]

            compliance_result = analyze_token_compliance(
                standard=token_standard_enum,
                has_profit_expectation=is_security_like or request.facts.get("has_profit_expectation", False),
                is_decentralized=request.facts.get("is_decentralized", not is_security_like),
                backed_by_fiat=is_stablecoin or request.facts.get("backed_by_fiat", False),
                # Howey test parameters
                investment_of_money=request.facts.get("investment_of_money", True),
                common_enterprise=request.facts.get("common_enterprise", is_security_like),
                efforts_of_promoter=request.facts.get("efforts_of_promoter", is_security_like),
                decentralization_score=request.facts.get("decentralization_score", 0.5 if not is_security_like else 0.1),
                is_functional_network=request.facts.get("is_functional_network", not is_security_like),
                # GENIUS Act parameters
                is_stablecoin=is_stablecoin,
                pegged_currency=request.facts.get("pegged_currency", "USD"),
                reserve_assets=request.facts.get("reserve_assets"),
                reserve_ratio=request.facts.get("reserve_ratio", 1.0),
                uses_algorithmic_mechanism=request.facts.get("uses_algorithmic_mechanism", False),
                issuer_charter_type=request.facts.get("issuer_charter_type", "non_bank_qualified"),
                has_reserve_attestation=request.facts.get("has_reserve_attestation", False),
                attestation_frequency_days=request.facts.get("attestation_frequency_days", 30),
            )
            token_compliance_result = {
                "standard": compliance_result.standard.value,
                "classification": compliance_result.classification.value,
                "requires_sec_registration": compliance_result.requires_sec_registration,
                "genius_act_applicable": compliance_result.genius_act_applicable,
                "sec_jurisdiction": compliance_result.sec_jurisdiction,
                "cftc_jurisdiction": compliance_result.cftc_jurisdiction,
                "compliance_requirements": compliance_result.compliance_requirements,
                "regulatory_risks": compliance_result.regulatory_risks,
                "recommended_actions": compliance_result.recommended_actions,
            }
            if compliance_result.howey_analysis:
                token_compliance_result["howey_analysis"] = {
                    "is_security": compliance_result.howey_analysis.is_security,
                    "investment_of_money": compliance_result.howey_analysis.investment_of_money,
                    "common_enterprise": compliance_result.howey_analysis.common_enterprise,
                    "expectation_of_profit": compliance_result.howey_analysis.expectation_of_profit,
                    "efforts_of_others": compliance_result.howey_analysis.efforts_of_others,
                    "analysis_notes": compliance_result.howey_analysis.analysis_notes,
                }
            if compliance_result.genius_analysis:
                token_compliance_result["genius_analysis"] = {
                    "is_compliant_stablecoin": compliance_result.genius_analysis.is_compliant_stablecoin,
                    "reserve_requirements_met": compliance_result.genius_analysis.reserve_requirements_met,
                    "issuer_requirements": compliance_result.genius_analysis.issuer_requirements,
                }
            audit_trail.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "TOKEN_COMPLIANCE_ANALYSIS",
                "details": {
                    "token_standard": request.token_standard,
                    "classification": compliance_result.classification.value,
                    "is_security": compliance_result.howey_analysis.is_security if compliance_result.howey_analysis else None,
                },
            })
        except Exception as e:
            audit_trail.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "TOKEN_COMPLIANCE_ERROR",
                "details": {"error": str(e)},
            })

    # Market Risk Enhancements: Protocol risk assessment
    protocol_risk_result = None
    if request.underlying_chain:
        chain_key = request.underlying_chain.lower()
        if chain_key in PROTOCOL_DEFAULTS:
            try:
                defaults = get_protocol_defaults(chain_key)
                assessment = assess_protocol_risk(protocol_id=chain_key, **defaults)
                protocol_risk_result = {
                    "protocol_id": assessment.protocol_id,
                    "risk_tier": assessment.risk_tier.value,
                    "overall_score": assessment.overall_score,
                    "consensus_score": assessment.consensus_score,
                    "decentralization_score": assessment.decentralization_score,
                    "settlement_score": assessment.settlement_score,
                    "operational_score": assessment.operational_score,
                    "security_score": assessment.security_score,
                    "risk_factors": assessment.risk_factors,
                    "strengths": assessment.strengths,
                    "regulatory_notes": assessment.regulatory_notes,
                }
                audit_trail.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "PROTOCOL_RISK_ASSESSMENT",
                    "details": {
                        "protocol": chain_key,
                        "risk_tier": assessment.risk_tier.value,
                        "overall_score": assessment.overall_score,
                    },
                })
            except Exception as e:
                audit_trail.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "PROTOCOL_RISK_ERROR",
                    "details": {"error": str(e)},
                })

    # Market Risk Enhancements: DeFi risk scoring
    defi_risk_result = None
    if request.is_defi_integrated and request.defi_protocol:
        protocol_key = request.defi_protocol.lower()
        if protocol_key in DEFI_PROTOCOL_DEFAULTS:
            try:
                defaults = DEFI_PROTOCOL_DEFAULTS[protocol_key]
                # Convert dict configurations to Pydantic models
                defi_score = score_defi_protocol(
                    protocol_id=protocol_key,
                    category=defaults["category"],
                    smart_contract=SmartContractRisk(**defaults["smart_contract"]),
                    economic=EconomicRisk(**defaults["economic"]),
                    oracle=OracleRisk(**defaults["oracle"]),
                    governance=GovernanceRisk(**defaults["governance"]),
                )
                defi_risk_result = {
                    "protocol_id": defi_score.protocol_id,
                    "category": defi_score.category.value,
                    "overall_grade": defi_score.overall_grade.value,
                    "overall_score": defi_score.overall_score,
                    "smart_contract_grade": defi_score.smart_contract_grade.value,
                    "smart_contract_score": defi_score.smart_contract_score,
                    "economic_grade": defi_score.economic_grade.value,
                    "economic_score": defi_score.economic_score,
                    "oracle_grade": defi_score.oracle_grade.value,
                    "oracle_score": defi_score.oracle_score,
                    "governance_grade": defi_score.governance_grade.value,
                    "governance_score": defi_score.governance_score,
                    "regulatory_flags": defi_score.regulatory_flags,
                    "critical_risks": defi_score.critical_risks,
                    "high_risks": defi_score.high_risks,
                    "strengths": defi_score.strengths,
                }
                audit_trail.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "DEFI_RISK_SCORING",
                    "details": {
                        "protocol": protocol_key,
                        "overall_grade": defi_score.overall_grade.value,
                        "overall_score": defi_score.overall_score,
                    },
                })
            except Exception as e:
                audit_trail.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "DEFI_RISK_ERROR",
                    "details": {"error": str(e)},
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
        token_compliance=token_compliance_result,
        protocol_risk=protocol_risk_result,
        defi_risk=defi_risk_result,
    )


@router.get("/jurisdictions")
async def list_jurisdictions() -> dict:
    """List all supported jurisdictions."""
    from backend.storage.database import get_db

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
    from backend.storage.database import get_db

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
    from backend.storage.database import get_db

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
