"""
Runtime package for KE Workbench.

Provides efficient rule execution using compiled IR.
"""

from backend.storage.retrieval.runtime.cache import IRCache, get_ir_cache, reset_ir_cache
from backend.storage.retrieval.runtime.trace import TraceStep, ExecutionTrace, DecisionResult
from backend.storage.retrieval.runtime.executor import RuleRuntime, execute_rule

__all__ = [
    # Cache
    "IRCache",
    "get_ir_cache",
    "reset_ir_cache",
    # Trace
    "TraceStep",
    "ExecutionTrace",
    "DecisionResult",
    # Executor
    "RuleRuntime",
    "execute_rule",
]
