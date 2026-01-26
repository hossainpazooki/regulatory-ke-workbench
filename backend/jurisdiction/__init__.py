"""Jurisdiction domain - cross-border compliance navigation."""

# Re-export router from existing location for backwards compatibility
from backend.core.api.routes_navigate import router
from .service import (
    # Constants
    DEFAULT_REGIMES,
    EXCLUSIVE_OBLIGATION_PAIRS,
    STEP_TIMELINES,
    STEP_DEPENDENCIES,
    # Resolver
    resolve_jurisdictions,
    get_equivalences,
    get_jurisdiction_info,
    get_regime_info,
    # Conflicts
    detect_conflicts,
    check_timeline_conflicts,
    # Pathway
    synthesize_pathway,
    aggregate_obligations,
    estimate_timeline,
    get_critical_path,
    # Evaluator
    evaluate_jurisdiction,
    evaluate_multiple_jurisdictions,
    evaluate_jurisdiction_sync,
)

__all__ = [
    # Router
    "router",
    # Constants
    "DEFAULT_REGIMES",
    "EXCLUSIVE_OBLIGATION_PAIRS",
    "STEP_TIMELINES",
    "STEP_DEPENDENCIES",
    # Resolver
    "resolve_jurisdictions",
    "get_equivalences",
    "get_jurisdiction_info",
    "get_regime_info",
    # Conflicts
    "detect_conflicts",
    "check_timeline_conflicts",
    # Pathway
    "synthesize_pathway",
    "aggregate_obligations",
    "estimate_timeline",
    "get_critical_path",
    # Evaluator
    "evaluate_jurisdiction",
    "evaluate_multiple_jurisdictions",
    "evaluate_jurisdiction_sync",
]
