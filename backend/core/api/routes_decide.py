"""Routes for regulatory decisions."""

from fastapi import APIRouter

from backend.core.ontology import Scenario
from backend.rules import RuleLoader, DecisionEngine
from backend.config import get_settings
from .models import (
    DecideRequest,
    DecideResponse,
    DecisionResponse,
    TraceStepResponse,
    ObligationResponse,
)

router = APIRouter(prefix="/decide", tags=["Decisions"])

# Global instances
_loader: RuleLoader | None = None
_engine: DecisionEngine | None = None


def get_loader() -> RuleLoader:
    """Get or create the rule loader instance."""
    global _loader
    if _loader is None:
        settings = get_settings()
        _loader = RuleLoader(settings.rules_dir)
        try:
            _loader.load_directory()
        except FileNotFoundError:
            pass  # No rules directory yet
    return _loader


def get_engine() -> DecisionEngine:
    """Get or create the decision engine instance."""
    global _engine
    if _engine is None:
        _engine = DecisionEngine(get_loader())
    return _engine


@router.post("", response_model=DecideResponse)
async def evaluate_scenario(request: DecideRequest) -> DecideResponse:
    """Evaluate a scenario against regulatory rules.

    Returns decision(s) with full trace showing how the conclusion was reached.
    """
    engine = get_engine()

    # Build scenario from request
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
        # Evaluate specific rule
        result = engine.evaluate(scenario, request.rule_id)
        results = [result]
    else:
        # Evaluate all applicable rules
        results = engine.evaluate_all(scenario)

    # Convert to response models
    responses = []
    for result in results:
        trace_steps = [
            TraceStepResponse(
                node=step.node,
                condition=step.condition,
                result=step.result,
                value_checked=step.value_checked,
            )
            for step in result.trace
        ]

        obligations = [
            ObligationResponse(
                id=obl.id,
                description=obl.description,
                source=obl.source,
                deadline=obl.deadline,
            )
            for obl in result.obligations
        ]

        responses.append(
            DecisionResponse(
                rule_id=result.rule_id,
                applicable=result.applicable,
                decision=result.decision,
                trace=trace_steps,
                obligations=obligations,
                source=result.source,
                notes=result.notes,
            )
        )

    # Generate summary
    summary = _generate_summary(responses)

    return DecideResponse(results=responses, summary=summary)


def _generate_summary(responses: list[DecisionResponse]) -> str | None:
    """Generate a summary of the decision results."""
    if not responses:
        return "No applicable rules found for this scenario."

    applicable = [r for r in responses if r.applicable]
    if not applicable:
        return "Scenario does not match any rule conditions."

    # Collect unique obligations
    all_obligations = []
    for r in applicable:
        all_obligations.extend(r.obligations)

    decisions = [r.decision for r in applicable if r.decision]
    unique_decisions = list(set(decisions))

    summary_parts = [f"Evaluated {len(applicable)} applicable rule(s)."]

    if unique_decisions:
        summary_parts.append(f"Outcomes: {', '.join(unique_decisions)}.")

    if all_obligations:
        summary_parts.append(f"{len(all_obligations)} obligation(s) triggered.")

    return " ".join(summary_parts)


@router.post("/reload")
async def reload_rules() -> dict:
    """Reload rules from disk."""
    global _loader, _engine
    _loader = None
    _engine = None

    loader = get_loader()
    rules = loader.get_all_rules()

    return {
        "status": "reloaded",
        "rules_loaded": len(rules),
    }
