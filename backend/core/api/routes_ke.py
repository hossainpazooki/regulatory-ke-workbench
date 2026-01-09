"""KE-internal API routes for Knowledge Engineering workbench.

These endpoints are for internal use by the KE team:
- Rule consistency verification
- Analytics and error patterns
- Drift detection
- Review queue management
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any

from datetime import datetime, timezone

from backend.rule_service.app.services import RuleLoader, DecisionEngine
from backend.rule_service.app.services.schema import (
    ConsistencyStatus,
    ConsistencyBlock,
    ConsistencySummary,
    ConsistencyEvidence,
)
from backend.verification_service.app.services import ConsistencyEngine, verify_rule
from backend.analytics_service.app.services import ErrorPatternAnalyzer, DriftDetector
from backend.rag_service.app.services import RuleContextRetriever
from backend.core.visualization import (
    build_rulebook_outline,
    build_decision_trace_tree,
    build_ontology_tree,
    build_corpus_rule_links,
    build_decision_tree_structure,
    is_supertree_available,
    render_rulebook_outline_html,
    render_decision_trace_html,
    render_ontology_tree_html,
    render_corpus_links_html,
)
from backend.core.ontology import Scenario


router = APIRouter(prefix="/ke", tags=["knowledge-engineering"])


# =============================================================================
# Request/Response Models
# =============================================================================

class VerifyRuleRequest(BaseModel):
    """Request to verify a rule."""
    rule_id: str
    source_text: str | None = None
    tiers: list[int] = Field(default=[0, 1])


class VerifyRuleResponse(BaseModel):
    """Response from rule verification."""
    rule_id: str
    status: str
    confidence: float
    evidence_count: int
    evidence: list[dict]


class AnalyticsSummaryResponse(BaseModel):
    """Analytics summary response."""
    total_rules: int
    verified: int
    needs_review: int
    inconsistent: int
    unverified: int
    verification_rate: float
    average_score: float
    timestamp: str


class ReviewQueueItem(BaseModel):
    """Item in review queue."""
    rule_id: str
    priority: float
    status: str
    confidence: float
    issues: list[str]


class ErrorPatternResponse(BaseModel):
    """Error pattern response."""
    pattern_id: str
    category: str
    description: str
    severity: str
    affected_rule_count: int
    affected_rules: list[str]
    recommendation: str


class DriftReportResponse(BaseModel):
    """Drift detection report."""
    report_id: str
    drift_detected: bool
    drift_severity: str
    degraded_categories: list[str]
    improved_categories: list[str]
    summary: str


class HumanReviewRequest(BaseModel):
    """Request to submit human review."""
    label: str = Field(..., description="Review decision: consistent, inconsistent, unknown")
    notes: str = Field(..., description="Reviewer notes explaining the decision")
    reviewer_id: str = Field(..., description="Identifier of the human reviewer")


class HumanReviewResponse(BaseModel):
    """Response from human review submission."""
    rule_id: str
    status: str
    confidence: float
    review_tier: int = 4
    reviewer_id: str
    message: str


# =============================================================================
# Shared State (would be dependency-injected in production)
# =============================================================================

# These are initialized lazily
_rule_loader: RuleLoader | None = None
_consistency_engine: ConsistencyEngine | None = None
_analyzer: ErrorPatternAnalyzer | None = None
_drift_detector: DriftDetector | None = None
_context_retriever: RuleContextRetriever | None = None


def get_rule_loader() -> RuleLoader:
    global _rule_loader
    if _rule_loader is None:
        from backend.core.config import get_settings
        settings = get_settings()
        _rule_loader = RuleLoader(settings.rules_dir)
        _rule_loader.load_directory()
    return _rule_loader


def get_consistency_engine() -> ConsistencyEngine:
    global _consistency_engine
    if _consistency_engine is None:
        _consistency_engine = ConsistencyEngine()
    return _consistency_engine


def get_analyzer() -> ErrorPatternAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ErrorPatternAnalyzer(rule_loader=get_rule_loader())
    return _analyzer


def get_drift_detector() -> DriftDetector:
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector(rule_loader=get_rule_loader())
    return _drift_detector


# =============================================================================
# Consistency Verification Endpoints
# =============================================================================

@router.post("/verify", response_model=VerifyRuleResponse)
def verify_rule_endpoint(request: VerifyRuleRequest):
    """Verify consistency of a single rule.

    Runs Tier 0 (structural) and Tier 1 (lexical) checks by default.
    """
    loader = get_rule_loader()
    rule = loader.get_rule(request.rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {request.rule_id}")

    engine = get_consistency_engine()
    result = engine.verify_rule(
        rule=rule,
        source_text=request.source_text,
        tiers=request.tiers,
    )

    return VerifyRuleResponse(
        rule_id=request.rule_id,
        status=result.summary.status.value,
        confidence=result.summary.confidence,
        evidence_count=len(result.evidence),
        evidence=[
            {
                "tier": ev.tier,
                "category": ev.category,
                "label": ev.label,
                "score": ev.score,
                "details": ev.details,
            }
            for ev in result.evidence
        ],
    )


@router.post("/verify-all")
def verify_all_rules(
    tiers: list[int] = Query(default=[0, 1]),
    save: bool = Query(default=False, description="Save results to rule files"),
) -> dict[str, Any]:
    """Verify all loaded rules.

    Returns summary and individual results.
    """
    loader = get_rule_loader()
    engine = get_consistency_engine()
    rules = loader.get_all_rules()

    results = []
    for rule in rules:
        consistency = engine.verify_rule(rule, tiers=tiers)
        results.append({
            "rule_id": rule.rule_id,
            "status": consistency.summary.status.value,
            "confidence": consistency.summary.confidence,
        })

        # Optionally save back to rule
        if save:
            rule.consistency = consistency

    return {
        "total": len(results),
        "verified": sum(1 for r in results if r["status"] == "verified"),
        "needs_review": sum(1 for r in results if r["status"] == "needs_review"),
        "inconsistent": sum(1 for r in results if r["status"] == "inconsistent"),
        "results": results,
    }


# =============================================================================
# Analytics Endpoints
# =============================================================================

@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
def get_analytics_summary():
    """Get summary statistics for all rules."""
    analyzer = get_analyzer()
    summary = analyzer.get_summary_stats()

    return AnalyticsSummaryResponse(**summary)


@router.get("/analytics/patterns", response_model=list[ErrorPatternResponse])
def get_error_patterns(min_affected: int = Query(default=2)):
    """Detect error patterns across rules."""
    analyzer = get_analyzer()
    patterns = analyzer.detect_patterns(min_affected=min_affected)

    return [
        ErrorPatternResponse(
            pattern_id=p.pattern_id,
            category=p.category,
            description=p.description,
            severity=p.severity,
            affected_rule_count=p.affected_rule_count,
            affected_rules=p.affected_rules,
            recommendation=p.recommendation,
        )
        for p in patterns
    ]


@router.get("/analytics/matrix")
def get_error_matrix() -> dict[str, dict[str, int]]:
    """Get error confusion matrix (category × outcome)."""
    analyzer = get_analyzer()
    return analyzer.build_error_matrix()


@router.get("/analytics/review-queue", response_model=list[ReviewQueueItem])
def get_review_queue(max_items: int = Query(default=50)):
    """Get prioritized review queue."""
    analyzer = get_analyzer()
    queue = analyzer.build_review_queue(max_items=max_items)

    return [
        ReviewQueueItem(
            rule_id=item.rule_id,
            priority=item.priority,
            status=item.status.value,
            confidence=item.confidence,
            issues=item.issues,
        )
        for item in queue
    ]


# =============================================================================
# Drift Detection Endpoints
# =============================================================================

@router.post("/drift/baseline")
def set_drift_baseline() -> dict:
    """Set current state as drift baseline."""
    detector = get_drift_detector()
    metrics = detector.set_baseline()

    return {
        "message": "Baseline set",
        "timestamp": metrics.timestamp,
        "total_rules": metrics.total_rules,
        "avg_confidence": metrics.avg_confidence,
    }


@router.get("/drift/detect", response_model=DriftReportResponse)
def detect_drift():
    """Detect drift from baseline."""
    detector = get_drift_detector()
    report = detector.detect_drift()

    return DriftReportResponse(
        report_id=report.report_id,
        drift_detected=report.drift_detected,
        drift_severity=report.drift_severity,
        degraded_categories=report.degraded_categories,
        improved_categories=report.improved_categories,
        summary=report.summary,
    )


@router.get("/drift/history")
def get_drift_history(window: int = Query(default=10)) -> list[dict]:
    """Get metrics history."""
    detector = get_drift_detector()
    history = detector.get_history()

    return [
        {
            "timestamp": m.timestamp,
            "total_rules": m.total_rules,
            "verified": m.verified_count,
            "avg_confidence": m.avg_confidence,
        }
        for m in history[-window:]
    ]


@router.get("/drift/authors")
def get_author_comparison() -> dict[str, Any]:
    """Compare consistency metrics by author."""
    detector = get_drift_detector()
    return detector.compare_authors()


# =============================================================================
# Rule Context Endpoints
# =============================================================================

@router.get("/context/{rule_id}")
def get_rule_context(rule_id: str) -> dict[str, Any]:
    """Get source context for a rule."""
    global _context_retriever

    loader = get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    if _context_retriever is None:
        _context_retriever = RuleContextRetriever(rule_loader=loader)

    context = _context_retriever.get_rule_context(rule)

    return {
        "rule_id": context.rule_id,
        "source_passages": [
            {"text": p.text[:200], "score": p.score, "document_id": p.document_id}
            for p in context.source_passages
        ],
        "cross_references": context.cross_references,
        "related_rules": context.related_rules,
    }


@router.get("/related/{rule_id}")
def get_related_rules(rule_id: str, top_k: int = Query(default=5)) -> list[dict]:
    """Get rules related to a given rule."""
    global _context_retriever

    loader = get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    if _context_retriever is None:
        _context_retriever = RuleContextRetriever(rule_loader=loader)

    related = _context_retriever.find_related_rules(rule, top_k=top_k)

    return [
        {
            "rule_id": r.rule_id,
            "description": r.description,
            "source": {
                "document_id": r.source.document_id if r.source else None,
                "article": r.source.article if r.source else None,
            } if r.source else None,
            "tags": r.tags,
        }
        for r in related
    ]


# =============================================================================
# Human Review Endpoints
# =============================================================================

@router.post("/rules/{rule_id}/review", response_model=HumanReviewResponse)
def submit_human_review(rule_id: str, request: HumanReviewRequest):
    """Submit a human review (Tier 4) for a rule.

    Human reviews are authoritative and override automated check labels.
    This appends a Tier 4 evidence item and updates the overall status.
    """
    if request.label not in ("consistent", "inconsistent", "unknown"):
        raise HTTPException(
            status_code=400,
            detail="Invalid label. Must be: consistent, inconsistent, unknown"
        )

    loader = get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    # Create Tier 4 human review evidence
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Map label to score
    score = {
        "consistent": 1.0,
        "inconsistent": 0.0,
        "unknown": 0.5,
    }.get(request.label, 0.5)

    human_evidence = ConsistencyEvidence(
        tier=4,
        category="human_review",
        label="pass" if request.label == "consistent" else (
            "fail" if request.label == "inconsistent" else "warning"
        ),
        score=score,
        details=f"Human review by {request.reviewer_id}: {request.notes}",
        rule_element="__rule__",
        timestamp=timestamp,
    )

    # Get existing consistency or create new
    if rule.consistency:
        existing_evidence = list(rule.consistency.evidence)
        existing_evidence.append(human_evidence)
    else:
        existing_evidence = [human_evidence]

    # Human review is authoritative - determine status based on human label
    new_status = {
        "consistent": ConsistencyStatus.VERIFIED,
        "inconsistent": ConsistencyStatus.INCONSISTENT,
        "unknown": ConsistencyStatus.NEEDS_REVIEW,
    }.get(request.label, ConsistencyStatus.NEEDS_REVIEW)

    # Calculate new confidence (weighted towards human review)
    if rule.consistency:
        # Average existing confidence with human score, weighted 60% human
        existing_conf = rule.consistency.summary.confidence
        new_confidence = (0.4 * existing_conf) + (0.6 * score)
    else:
        new_confidence = score

    # Create updated consistency block
    new_summary = ConsistencySummary(
        status=new_status,
        confidence=round(new_confidence, 4),
        last_verified=timestamp,
        verified_by=f"human:{request.reviewer_id}",
        notes=request.notes,
    )

    rule.consistency = ConsistencyBlock(
        summary=new_summary,
        evidence=existing_evidence,
    )

    return HumanReviewResponse(
        rule_id=rule_id,
        status=new_status.value,
        confidence=new_confidence,
        review_tier=4,
        reviewer_id=request.reviewer_id,
        message=f"Human review submitted. Status updated to {new_status.value}.",
    )


@router.get("/rules/{rule_id}/reviews")
def get_rule_reviews(rule_id: str) -> list[dict]:
    """Get all human reviews for a rule."""
    loader = get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    if not rule.consistency:
        return []

    # Filter for Tier 4 human reviews
    reviews = [
        {
            "tier": ev.tier,
            "category": ev.category,
            "label": ev.label,
            "score": ev.score,
            "details": ev.details,
            "timestamp": ev.timestamp,
        }
        for ev in rule.consistency.evidence
        if ev.tier == 4 and ev.category == "human_review"
    ]

    return reviews


# =============================================================================
# Chart Visualization Endpoints
# =============================================================================

class ChartDataResponse(BaseModel):
    """Response containing tree data for visualization."""
    chart_type: str
    data: dict
    supertree_available: bool


class ChartHtmlResponse(BaseModel):
    """Response containing rendered HTML chart."""
    chart_type: str
    html: str
    supertree_available: bool


@router.get("/charts/supertree-status")
def get_supertree_status() -> dict:
    """Check if Supertree visualization is available."""
    return {
        "available": is_supertree_available(),
        "message": (
            "Supertree is available for interactive charts"
            if is_supertree_available()
            else "Install supertree for interactive charts: pip install -r requirements-visualization.txt"
        ),
    }


@router.get("/charts/rulebook-outline", response_model=ChartDataResponse)
def get_rulebook_outline_chart():
    """Get rulebook outline tree data for visualization."""
    loader = get_rule_loader()
    rules = loader.get_all_rules()
    tree_data = build_rulebook_outline(rules)

    return ChartDataResponse(
        chart_type="rulebook_outline",
        data=tree_data,
        supertree_available=is_supertree_available(),
    )


@router.get("/charts/rulebook-outline/html", response_model=ChartHtmlResponse)
def get_rulebook_outline_html():
    """Get rulebook outline as rendered HTML."""
    loader = get_rule_loader()
    rules = loader.get_all_rules()
    tree_data = build_rulebook_outline(rules)
    html = render_rulebook_outline_html(tree_data)

    return ChartHtmlResponse(
        chart_type="rulebook_outline",
        html=html,
        supertree_available=is_supertree_available(),
    )


@router.get("/charts/ontology", response_model=ChartDataResponse)
def get_ontology_chart():
    """Get ontology tree data for visualization."""
    tree_data = build_ontology_tree()

    return ChartDataResponse(
        chart_type="ontology",
        data=tree_data,
        supertree_available=is_supertree_available(),
    )


@router.get("/charts/ontology/html", response_model=ChartHtmlResponse)
def get_ontology_html():
    """Get ontology tree as rendered HTML."""
    tree_data = build_ontology_tree()
    html = render_ontology_tree_html(tree_data)

    return ChartHtmlResponse(
        chart_type="ontology",
        html=html,
        supertree_available=is_supertree_available(),
    )


@router.get("/charts/corpus-links", response_model=ChartDataResponse)
def get_corpus_links_chart():
    """Get corpus-to-rule links tree data for visualization."""
    loader = get_rule_loader()
    rules = loader.get_all_rules()
    tree_data = build_corpus_rule_links(rules)

    return ChartDataResponse(
        chart_type="corpus_links",
        data=tree_data,
        supertree_available=is_supertree_available(),
    )


@router.get("/charts/corpus-links/html", response_model=ChartHtmlResponse)
def get_corpus_links_html():
    """Get corpus-to-rule links as rendered HTML."""
    loader = get_rule_loader()
    rules = loader.get_all_rules()
    tree_data = build_corpus_rule_links(rules)
    html = render_corpus_links_html(tree_data)

    return ChartHtmlResponse(
        chart_type="corpus_links",
        html=html,
        supertree_available=is_supertree_available(),
    )


@router.get("/charts/decision-tree/{rule_id}", response_model=ChartDataResponse)
def get_decision_tree_chart(rule_id: str):
    """Get decision tree structure for a specific rule."""
    loader = get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    if rule.decision_tree is None:
        raise HTTPException(
            status_code=404,
            detail=f"Rule {rule_id} has no decision tree"
        )

    tree_data = build_decision_tree_structure(rule.decision_tree)

    return ChartDataResponse(
        chart_type="decision_tree",
        data=tree_data or {},
        supertree_available=is_supertree_available(),
    )


class EvaluateForTraceRequest(BaseModel):
    """Request to evaluate a rule and get decision trace."""
    scenario: dict = Field(..., description="Scenario attributes as key-value pairs")


@router.post("/charts/decision-trace/{rule_id}", response_model=ChartDataResponse)
def get_decision_trace_chart(rule_id: str, request: EvaluateForTraceRequest):
    """Evaluate a rule against a scenario and return decision trace tree.

    This endpoint evaluates the rule and returns the trace of the evaluation
    path through the decision tree.
    """
    loader = get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    # Create scenario from request
    scenario = Scenario(**request.scenario)

    # Evaluate rule
    engine = DecisionEngine(loader)
    result = engine.evaluate(scenario, rule_id)

    # Build trace tree
    tree_data = build_decision_trace_tree(
        trace=result.trace,
        decision=result.decision,
        rule_id=rule_id,
    )

    return ChartDataResponse(
        chart_type="decision_trace",
        data=tree_data,
        supertree_available=is_supertree_available(),
    )


@router.post("/charts/decision-trace/{rule_id}/html", response_model=ChartHtmlResponse)
def get_decision_trace_html(rule_id: str, request: EvaluateForTraceRequest):
    """Evaluate a rule against a scenario and return decision trace as HTML."""
    loader = get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    # Create scenario from request
    scenario = Scenario(**request.scenario)

    # Evaluate rule
    engine = DecisionEngine(loader)
    result = engine.evaluate(scenario, rule_id)

    # Build trace tree and render
    tree_data = build_decision_trace_tree(
        trace=result.trace,
        decision=result.decision,
        rule_id=rule_id,
    )
    html = render_decision_trace_html(tree_data)

    return ChartHtmlResponse(
        chart_type="decision_trace",
        html=html,
        supertree_available=is_supertree_available(),
    )
