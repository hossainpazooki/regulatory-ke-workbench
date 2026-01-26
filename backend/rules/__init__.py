"""Rules domain - decision engine and rule inspection."""

from .router import decide_router, rules_router, get_loader, get_engine
from .schemas import (
    DecideRequest,
    DecideResponse,
    DecisionResponse,
    TraceStepResponse,
    ObligationResponse,
    RulesListResponse,
    RuleInfo,
    RuleDetailResponse,
    RuleVersionResponse,
    RuleVersionListResponse,
    RuleVersionDetailResponse,
    RuleEventResponse,
    RuleEventListResponse,
)
from .service import (
    # Models
    SourceRef,
    ComparisonOp,
    ConditionSpec,
    ConditionGroupSpec,
    ObligationSpec,
    ObligationResult,
    DecisionLeaf,
    DecisionNode,
    DecisionBranch,  # Alias for DecisionNode (backwards compatibility)
    ConsistencyStatus,
    ConsistencyLabel,
    ConsistencyEvidence,
    ConsistencySummary,
    ConsistencyBlock,
    Rule,
    RulePack,
    TraceStep,
    RuleMetadata,
    DecisionResult,
    # Services
    RuleLoader,
    DecisionEngine,
)
from .jurisdiction import (
    # Resolver
    resolve_jurisdictions,
    get_equivalences,
    get_jurisdiction_info,
    get_regime_info,
    DEFAULT_REGIMES,
    # Conflicts
    detect_conflicts,
    check_timeline_conflicts,
    EXCLUSIVE_OBLIGATION_PAIRS,
    # Pathway
    synthesize_pathway,
    aggregate_obligations,
    estimate_timeline,
    get_critical_path,
    STEP_TIMELINES,
    STEP_DEPENDENCIES,
    # Evaluator
    evaluate_jurisdiction,
    evaluate_multiple_jurisdictions,
    evaluate_jurisdiction_sync,
)

__all__ = [
    # Routers
    "decide_router",
    "rules_router",
    # Service functions
    "get_loader",
    "get_engine",
    # API Schemas
    "DecideRequest",
    "DecideResponse",
    "DecisionResponse",
    "TraceStepResponse",
    "ObligationResponse",
    "RulesListResponse",
    "RuleInfo",
    "RuleDetailResponse",
    "RuleVersionResponse",
    "RuleVersionListResponse",
    "RuleVersionDetailResponse",
    "RuleEventResponse",
    "RuleEventListResponse",
    # Service Models
    "SourceRef",
    "ComparisonOp",
    "ConditionSpec",
    "ConditionGroupSpec",
    "ObligationSpec",
    "ObligationResult",
    "DecisionLeaf",
    "DecisionNode",
    "ConsistencyStatus",
    "ConsistencyLabel",
    "ConsistencyEvidence",
    "ConsistencySummary",
    "ConsistencyBlock",
    "Rule",
    "RulePack",
    "TraceStep",
    "RuleMetadata",
    "DecisionResult",
    # Services
    "RuleLoader",
    "DecisionEngine",
    # Legacy schema exports
    "DecisionBranch",
    # Jurisdiction - Resolver
    "resolve_jurisdictions",
    "get_equivalences",
    "get_jurisdiction_info",
    "get_regime_info",
    "DEFAULT_REGIMES",
    # Jurisdiction - Conflicts
    "detect_conflicts",
    "check_timeline_conflicts",
    "EXCLUSIVE_OBLIGATION_PAIRS",
    # Jurisdiction - Pathway
    "synthesize_pathway",
    "aggregate_obligations",
    "estimate_timeline",
    "get_critical_path",
    "STEP_TIMELINES",
    "STEP_DEPENDENCIES",
    # Jurisdiction - Evaluator
    "evaluate_jurisdiction",
    "evaluate_multiple_jurisdictions",
    "evaluate_jurisdiction_sync",
]
