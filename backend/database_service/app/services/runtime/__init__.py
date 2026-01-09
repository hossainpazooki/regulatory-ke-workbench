"""
Runtime package for KE Workbench.

Provides efficient rule execution using compiled IR.
"""

from backend.database_service.app.services.runtime.cache import IRCache, get_ir_cache, reset_ir_cache
from backend.database_service.app.services.runtime.trace import TraceStep, ExecutionTrace, DecisionResult
from backend.database_service.app.services.runtime.executor import RuleRuntime, execute_rule

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
