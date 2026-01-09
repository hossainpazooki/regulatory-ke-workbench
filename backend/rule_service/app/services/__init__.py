"""Rule service - rule management, decision engine, and jurisdiction."""

from .loader import (
    RuleLoader,
    Rule,
    SourceRef,
    ConditionSpec,
    ConditionGroupSpec,
    DecisionNode,
    DecisionLeaf,
    ObligationSpec,
)
from .engine import DecisionEngine, TraceStep, ObligationResult
from .schema import (
    ConsistencyBlock,
    ConsistencySummary,
    ConsistencyEvidence,
    ConsistencyStatus,
    DecisionBranch,
    ConditionGroup,
)

__all__ = [
    # Loader
    "RuleLoader",
    "Rule",
    "SourceRef",
    "ConditionSpec",
    "ConditionGroupSpec",
    "DecisionNode",
    "DecisionLeaf",
    "ObligationSpec",
    # Engine
    "DecisionEngine",
    "TraceStep",
    "ObligationResult",
    # Schema
    "ConsistencyBlock",
    "ConsistencySummary",
    "ConsistencyEvidence",
    "ConsistencyStatus",
    "DecisionBranch",
    "ConditionGroup",
]
