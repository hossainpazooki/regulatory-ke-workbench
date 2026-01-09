"""
Compiler package for KE Workbench.

Provides compile-time transformation of rules to Intermediate Representation (IR)
for O(1) rule lookup and linear condition evaluation at runtime.
"""

from backend.database_service.app.services.compiler.ir import (
    CompiledCheck,
    DecisionEntry,
    ObligationSpec,
    RuleIR,
)
from backend.database_service.app.services.compiler.compiler import RuleCompiler, compile_rule, compile_rules
from backend.database_service.app.services.compiler.premise_index import PremiseIndexBuilder, get_premise_index

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
    # Index
    "PremiseIndexBuilder",
    "get_premise_index",
]
