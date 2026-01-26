"""Routes for counterfactual analysis service."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.decoder import (
    CounterfactualEngine,
    CounterfactualResponse,
    ComparisonMatrix,
    Scenario,
    ScenarioType,
    ExplanationTier,
)
from backend.rules import RuleLoader, DecisionEngine
from backend.core.config import get_settings

router = APIRouter(prefix="/counterfactual", tags=["Counterfactual"])

# Global instances
_loader: RuleLoader | None = None
_engine: DecisionEngine | None = None
_counterfactual: CounterfactualEngine | None = None


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


def get_counterfactual() -> CounterfactualEngine:
    """Get or create the counterfactual engine instance."""
    global _counterfactual
    if _counterfactual is None:
        _counterfactual = CounterfactualEngine(
            decision_engine=get_engine(),
        )
    return _counterfactual


# Request models


class ScenarioRequest(BaseModel):
    """A counterfactual scenario definition."""

    type: ScenarioType
    name: str | None = None
    parameters: dict[str, Any] = {}


class AnalyzeByIdRequest(BaseModel):
    """Request to analyze counterfactual for existing decision."""

    baseline_decision_id: str
    scenario: ScenarioRequest
    include_explanation: bool = True
    explanation_tier: ExplanationTier = ExplanationTier.INSTITUTIONAL


class InlineAnalyzeRequest(BaseModel):
    """Request with inline scenario for immediate counterfactual analysis."""

    # Baseline scenario fields
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

    # Evaluation options
    rule_id: str | None = None

    # Counterfactual scenario
    scenario: ScenarioRequest

    # Options
    include_explanation: bool = True
    explanation_tier: ExplanationTier = ExplanationTier.INSTITUTIONAL


class CompareByIdRequest(BaseModel):
    """Request to compare multiple scenarios for existing decision."""

    baseline_decision_id: str
    scenarios: list[ScenarioRequest]


class InlineCompareRequest(BaseModel):
    """Request with inline scenario for multi-scenario comparison."""

    # Baseline scenario fields
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

    # Evaluation options
    rule_id: str | None = None

    # Counterfactual scenarios
    scenarios: list[ScenarioRequest]


@router.post("/analyze", response_model=CounterfactualResponse)
async def analyze_counterfactual(
    request: AnalyzeByIdRequest,
) -> CounterfactualResponse:
    """Analyze a counterfactual scenario for an existing decision.

    This endpoint looks up a decision by ID and analyzes how a scenario change
    would affect the compliance outcome.

    Note: Currently returns a placeholder as decision storage is not yet implemented.
    """
    cf_engine = get_counterfactual()

    scenario = Scenario(
        type=request.scenario.type,
        name=request.scenario.name,
        parameters=request.scenario.parameters,
    )

    return cf_engine.analyze_by_id(
        request=type(
            "CounterfactualRequest",
            (),
            {
                "baseline_decision_id": request.baseline_decision_id,
                "scenario": scenario,
                "include_explanation": request.include_explanation,
                "explanation_tier": request.explanation_tier,
            },
        )()
    )


@router.post("/analyze/inline", response_model=CounterfactualResponse)
async def analyze_inline(request: InlineAnalyzeRequest) -> CounterfactualResponse:
    """Evaluate baseline and analyze counterfactual in one call.

    This endpoint:
    1. Evaluates the baseline scenario against rules
    2. Applies the counterfactual scenario
    3. Compares outcomes and generates delta analysis
    """
    from backend.core.ontology import Scenario as OntologyScenario

    engine = get_engine()
    cf_engine = get_counterfactual()

    # Build baseline scenario
    baseline_scenario = OntologyScenario(
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

    # Evaluate baseline
    if request.rule_id:
        baseline_result = engine.evaluate(baseline_scenario, request.rule_id)
    else:
        results = engine.evaluate_all(baseline_scenario)
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No applicable rules found for baseline scenario",
            )
        baseline_result = next((r for r in results if r.applicable), results[0])

    # Build counterfactual scenario
    cf_scenario = Scenario(
        type=request.scenario.type,
        name=request.scenario.name,
        parameters=request.scenario.parameters,
    )

    # Analyze counterfactual
    return cf_engine.analyze(
        baseline_decision=baseline_result,
        scenario=cf_scenario,
        include_explanation=request.include_explanation,
        explanation_tier=request.explanation_tier,
    )


@router.post("/compare", response_model=ComparisonMatrix)
async def compare_scenarios(request: CompareByIdRequest) -> ComparisonMatrix:
    """Compare multiple counterfactual scenarios for an existing decision.

    Note: Currently returns a placeholder as decision storage is not yet implemented.
    """
    cf_engine = get_counterfactual()

    scenarios = [
        Scenario(
            type=s.type,
            name=s.name,
            parameters=s.parameters,
        )
        for s in request.scenarios
    ]

    return cf_engine.compare_by_id(
        request=type(
            "ComparisonRequest",
            (),
            {
                "baseline_decision_id": request.baseline_decision_id,
                "scenarios": scenarios,
            },
        )()
    )


@router.post("/compare/inline", response_model=ComparisonMatrix)
async def compare_inline(request: InlineCompareRequest) -> ComparisonMatrix:
    """Evaluate baseline and compare multiple scenarios in one call.

    This endpoint:
    1. Evaluates the baseline scenario
    2. Applies each counterfactual scenario
    3. Builds a comparison matrix with insights
    """
    from backend.core.ontology import Scenario as OntologyScenario

    engine = get_engine()
    cf_engine = get_counterfactual()

    # Build baseline scenario
    baseline_scenario = OntologyScenario(
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

    # Evaluate baseline
    if request.rule_id:
        baseline_result = engine.evaluate(baseline_scenario, request.rule_id)
    else:
        results = engine.evaluate_all(baseline_scenario)
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No applicable rules found for baseline scenario",
            )
        baseline_result = next((r for r in results if r.applicable), results[0])

    # Build counterfactual scenarios
    cf_scenarios = [
        Scenario(
            type=s.type,
            name=s.name,
            parameters=s.parameters,
        )
        for s in request.scenarios
    ]

    # Compare scenarios
    return cf_engine.compare(
        baseline_decision=baseline_result,
        scenarios=cf_scenarios,
    )


@router.get("/scenario-types")
async def list_scenario_types() -> list[dict]:
    """List available counterfactual scenario types."""
    return [
        {
            "id": "jurisdiction_change",
            "name": "Jurisdiction Change",
            "description": "What if the activity was in a different jurisdiction?",
            "parameters": ["from_jurisdiction", "to_jurisdiction"],
        },
        {
            "id": "entity_change",
            "name": "Entity Type Change",
            "description": "What if the entity type was different?",
            "parameters": ["from_entity_type", "to_entity_type"],
        },
        {
            "id": "activity_restructure",
            "name": "Activity Restructure",
            "description": "What if the activity was structured differently?",
            "parameters": ["new_activity", "modifications"],
        },
        {
            "id": "threshold",
            "name": "Threshold Change",
            "description": "What if certain thresholds changed?",
            "parameters": ["threshold_type", "new_value"],
        },
        {
            "id": "temporal",
            "name": "Temporal Change",
            "description": "What if evaluated at a different time?",
            "parameters": ["effective_date"],
        },
        {
            "id": "protocol_change",
            "name": "Protocol Change",
            "description": "What if using a different protocol/technology?",
            "parameters": ["protocol", "technical_features"],
        },
        {
            "id": "regulatory_change",
            "name": "Regulatory Change",
            "description": "What if regulations changed?",
            "parameters": ["change_type", "new_requirements"],
        },
    ]
