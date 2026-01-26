"""Regulatory KE Workbench - Computational law platform for tokenized RWAs.

This module provides the core regulatory reasoning engine.
All components are pure Python - no OCaml required at runtime.

Environment Variables:
    USE_OCAML_ENGINE: Set to "true" to use OCaml engine (requires compilation).
                      Default is "false" (pure Python).
"""

import os

# Feature flag for engine selection
# Streamlit Cloud and other Python-only environments use the Python engine
USE_OCAML = os.getenv("USE_OCAML_ENGINE", "false").lower() == "true"

if USE_OCAML:
    # OCaml engine would be used if compiled and available
    # Currently falls back to Python
    ENGINE_TYPE = "python_fallback"
else:
    ENGINE_TYPE = "python"

# Core ontology types
from .core.ontology import (
    Actor,
    ActorType,
    Activity,
    ActivityType,
    Condition,
    ConditionGroup,
    Instrument,
    InstrumentType,
    NormativeContent,
    Obligation,
    Permission,
    Prohibition,
    Provision,
    ProvisionType,
    Relation,
    RelationType,
    Scenario,
    SourceReference,
)

# Rules and decision engine
from .rules import (
    DecisionEngine,
    Rule,
    RuleLoader,
    TraceStep,
)

# Verification engine
from .verification import (
    ConsistencyEngine,
    verify_rule,
)

__version__ = "0.1.0"

__all__ = [
    # Engine configuration
    "USE_OCAML",
    "ENGINE_TYPE",
    # Ontology types
    "Actor",
    "ActorType",
    "Activity",
    "ActivityType",
    "Condition",
    "ConditionGroup",
    "Instrument",
    "InstrumentType",
    "NormativeContent",
    "Obligation",
    "Permission",
    "Prohibition",
    "Provision",
    "ProvisionType",
    "Relation",
    "RelationType",
    "Scenario",
    "SourceReference",
    # Rules
    "DecisionEngine",
    "Rule",
    "RuleLoader",
    "TraceStep",
    # Verification
    "ConsistencyEngine",
    "verify_rule",
]
