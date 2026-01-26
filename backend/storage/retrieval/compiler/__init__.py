"""
Compiler package for KE Workbench.

Provides compile-time transformation of rules to Intermediate Representation (IR)
for O(1) rule lookup and linear condition evaluation at runtime.
"""

from backend.storage.retrieval.compiler.ir import (
    CompiledCheck,
    DecisionEntry,
    ObligationSpec,
    RuleIR,
)
from backend.storage.retrieval.compiler.compiler import RuleCompiler, compile_rule, compile_rules
from backend.storage.retrieval.compiler.optimizer import RuleOptimizer, optimize_rule
from backend.storage.retrieval.compiler.premise_index import PremiseIndexBuilder, get_premise_index

__all__ = [
    # IR Types
    "CompiledCheck",
    "DecisionEntry",
    "ObligationSpec",
    "RuleIR",
    # Compiler
    "RuleCompiler",
    "compile_rule",
    "compile_rules",
    # Optimizer
    "RuleOptimizer",
    "optimize_rule",
    # Index
    "PremiseIndexBuilder",
    "get_premise_index",
]
