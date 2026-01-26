"""
Retrieval Engine package for KE Workbench.

Provides compile-time rule transformation and efficient runtime execution.
Combines compiler/ for IR generation and runtime/ for execution.
"""

# Re-export from compiler
from backend.storage.retrieval.compiler import (
    # IR Types
    CompiledCheck,
    DecisionEntry,
    ObligationSpec,
    RuleIR,
    # Compiler
    RuleCompiler,
    compile_rule,
    compile_rules,
    # Optimizer
    RuleOptimizer,
    optimize_rule,
    # Index
    PremiseIndexBuilder,
    get_premise_index,
)

# Re-export from runtime
from backend.storage.retrieval.runtime import (
    # Cache
    IRCache,
    get_ir_cache,
    reset_ir_cache,
    # Trace
    TraceStep,
    ExecutionTrace,
    DecisionResult,
    # Executor
    RuleRuntime,
    execute_rule,
)

__all__ = [
    # Compiler - IR Types
    "CompiledCheck",
    "DecisionEntry",
    "ObligationSpec",
    "RuleIR",
    # Compiler - Functions
    "RuleCompiler",
    "compile_rule",
    "compile_rules",
    # Compiler - Optimizer
    "RuleOptimizer",
    "optimize_rule",
    # Compiler - Index
    "PremiseIndexBuilder",
    "get_premise_index",
    # Runtime - Cache
    "IRCache",
    "get_ir_cache",
    "reset_ir_cache",
    # Runtime - Trace
    "TraceStep",
    "ExecutionTrace",
    "DecisionResult",
    # Runtime - Executor
    "RuleRuntime",
    "execute_rule",
]
