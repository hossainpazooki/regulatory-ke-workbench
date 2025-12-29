"""
Runtime package for KE Workbench.

Provides efficient IR-based rule execution with:
- O(1) rule lookup via premise index
- Linear condition evaluation
- Decision table lookup
- Execution tracing for auditability
"""

from backend.runtime.executor import RuleRuntime, execute_rule
from backend.runtime.cache import IRCache, get_ir_cache
from backend.runtime.trace import ExecutionTrace, TraceStep

__all__ = [
    # Executor
    "RuleRuntime",
    "execute_rule",
    # Cache
    "IRCache",
    "get_ir_cache",
    # Trace
    "ExecutionTrace",
    "TraceStep",
]
