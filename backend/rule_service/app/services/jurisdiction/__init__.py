"""
Jurisdiction services for cross-border compliance.

Provides jurisdiction resolution, conflict detection, pathway synthesis,
and parallel multi-jurisdiction evaluation.
"""

from .resolver import (
    resolve_jurisdictions,
    get_equivalences,
    get_jurisdiction_info,
    get_regime_info,
    DEFAULT_REGIMES,
)
from .conflicts import (
    detect_conflicts,
    check_timeline_conflicts,
    EXCLUSIVE_OBLIGATION_PAIRS,
)
from .pathway import (
    synthesize_pathway,
    aggregate_obligations,
    estimate_timeline,
    get_critical_path,
    STEP_TIMELINES,
    STEP_DEPENDENCIES,
)
from .evaluator import (
    evaluate_jurisdiction,
    evaluate_multiple_jurisdictions,
    evaluate_jurisdiction_sync,
)

__all__ = [
    # Resolver
    "resolve_jurisdictions",
    "get_equivalences",
    "get_jurisdiction_info",
    "get_regime_info",
    "DEFAULT_REGIMES",
    # Conflicts
    "detect_conflicts",
    "check_timeline_conflicts",
    "EXCLUSIVE_OBLIGATION_PAIRS",
    # Pathway
    "synthesize_pathway",
    "aggregate_obligations",
    "estimate_timeline",
    "get_critical_path",
    "STEP_TIMELINES",
    "STEP_DEPENDENCIES",
    # Evaluator
    "evaluate_jurisdiction",
    "evaluate_multiple_jurisdictions",
    "evaluate_jurisdiction_sync",
]
