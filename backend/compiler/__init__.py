"""
Compiler package for KE Workbench.

Provides compile-time transformation of rules to Intermediate Representation (IR)
for O(1) rule lookup and linear condition evaluation at runtime.
"""

from backend.compiler.ir import (
    CompiledCheck,
    DecisionEntry,
    ObligationSpec,
    RuleIR,
)
from backend.compiler.compiler import RuleCompiler
from backend.compiler.premise_index import PremiseIndexBuilder

__all__ = [
    # IR Types
    "CompiledCheck",
    "DecisionEntry",
    "ObligationSpec",
    "RuleIR",
    # Compiler
    "RuleCompiler",
    # Index
    "PremiseIndexBuilder",
]
