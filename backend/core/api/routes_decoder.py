"""Routes for explanation decoder service."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.decoder import (
    DecoderService,
    DecoderResponse,
    ExplanationTier,
)
from backend.rules import RuleLoader, DecisionEngine
from backend.core.config import get_settings

router = APIRouter(prefix="/decoder", tags=["Decoder"])

# Global instances
_loader: RuleLoader | None = None
_engine: DecisionEngine | None = None
_decoder: DecoderService | None = None


def get_loader() -> RuleLoader:
    """Get or create the rule loader instance."""
    global _loader
    if _loader is None:
        settings = get_settings()
        _loader = RuleLoader(settings.rules_dir)
        try:
            _loader.load_directory()
        except FileNotFoundError:
            pass
    return _loader


def get_engine() -> DecisionEngine:
    """Get or create the decision engine instance."""
    global _engine
    if _engine is None:
        _engine = DecisionEngine(get_loader())
    return _engine


def get_decoder() -> DecoderService:
    """Get or create the decoder service instance."""
    global _decoder
    if _decoder is None:
        _decoder = DecoderService()
    return _decoder


class TemplateInfo(BaseModel):
    """Template summary for listing."""

    id: str
    name: str
    version: str
    frameworks: list[str]
    activity_types: list[str]
    outcome: str


class ExplainByDecisionRequest(BaseModel):
    """Request to explain an existing decision result."""

    decision_id: str
    tier: ExplanationTier = ExplanationTier.INSTITUTIONAL
    include_citations: bool = True


class InlineDecisionRequest(BaseModel):
    """Request with inline scenario for immediate evaluation and explanation."""

    # Scenario fields
    instrument_type: str | None = None
    activity: str | None = None
    jurisdiction: str | None = None
    authorized: bool | None = None
    actor_type: str | None = None
    issuer_type: str | None = None
    is_credit_institution: bool | None = None
    is_authorized_institution: bool | None = None
    reference_asset: str | None = None
    is_significant: bool | None = None
    reserve_value_eur: float | None = None
    extra: dict[str, Any] = {}

    # Decoder options
    rule_id: str | None = None
    tier: ExplanationTier = ExplanationTier.INSTITUTIONAL
    include_citations: bool = True


@router.post("/explain", response_model=DecoderResponse)
async def explain_decision(request: ExplainByDecisionRequest) -> DecoderResponse:
    """Generate tiered explanation for an existing decision.

    This endpoint looks up a decision by ID and generates an explanation.
    Note: Currently returns a placeholder as decision storage is not yet implemented.
    """
    decoder = get_decoder()
    return decoder.explain_by_id(
        decision_id=request.decision_id,
        tier=request.tier,
        include_citations=request.include_citations,
    )


@router.post("/explain/inline", response_model=DecoderResponse)
async def explain_inline(request: InlineDecisionRequest) -> DecoderResponse:
    """Evaluate a scenario and generate explanation in one call.

    This endpoint:
    1. Evaluates the scenario against rules
    2. Generates a tiered explanation for the result
    """
    from backend.core.ontology import Scenario

    engine = get_engine()
    decoder = get_decoder()

    # Build scenario
    scenario = Scenario(
        instrument_type=request.instrument_type,
        activity=request.activity,
        jurisdiction=request.jurisdiction,
        authorized=request.authorized,
        actor_type=request.actor_type,
        issuer_type=request.issuer_type,
        is_credit_institution=request.is_credit_institution,
        is_authorized_institution=request.is_authorized_institution,
        reference_asset=request.reference_asset,
        is_significant=request.is_significant,
        reserve_value_eur=request.reserve_value_eur,
        extra=request.extra,
    )

    # Evaluate
    if request.rule_id:
        result = engine.evaluate(scenario, request.rule_id)
    else:
        results = engine.evaluate_all(scenario)
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No applicable rules found for this scenario",
            )
        # Use first applicable result
        result = next((r for r in results if r.applicable), results[0])

    # Generate explanation
    return decoder.explain(
        decision=result,
        tier=request.tier,
        include_citations=request.include_citations,
    )


@router.get("/templates")
async def list_templates() -> list[TemplateInfo]:
    """List available explanation templates."""
    decoder = get_decoder()
    templates = decoder.templates.list_templates()

    return [
        TemplateInfo(
            id=t.id,
            name=t.name,
            version=t.version,
            frameworks=t.frameworks,
            activity_types=t.activity_types,
            outcome=t.outcome,
        )
        for t in templates
    ]


@router.get("/templates/{template_id}")
async def get_template(template_id: str) -> dict:
    """Get details of a specific template."""
    decoder = get_decoder()
    template = decoder.templates.get(template_id)

    if not template:
        raise HTTPException(
            status_code=404,
            detail=f"Template not found: {template_id}",
        )

    return {
        "id": template.id,
        "name": template.name,
        "version": template.version,
        "frameworks": template.frameworks,
        "activity_types": template.activity_types,
        "outcome": template.outcome,
        "tiers": {
            tier.value: [
                {"type": s.type, "template": s.template, "llm_enhance": s.llm_enhance}
                for s in sections
            ]
            for tier, sections in template.tiers.items()
        },
        "variables": [
            {"name": v.name, "source": v.source, "required": v.required}
            for v in template.variables
        ],
    }


@router.get("/tiers")
async def list_tiers() -> list[dict]:
    """List available explanation tiers."""
    return [
        {
            "id": "retail",
            "name": "Retail",
            "description": "Plain language, 2-3 sentences for end users",
        },
        {
            "id": "protocol",
            "name": "Protocol",
            "description": "Technical rationale for smart contract integration",
        },
        {
            "id": "institutional",
            "name": "Institutional",
            "description": "Full compliance report for institutions",
        },
        {
            "id": "regulator",
            "name": "Regulator",
            "description": "Complete legal analysis for regulatory review",
        },
    ]
