"""Pydantic schemas for embedding rule API.

Re-exports from backend.rule_embedding_service.app.services.schemas for backwards compatibility during migration.
"""
from backend.rule_embedding_service.app.services.schemas import (
    ConditionCreate,
    ConditionRead,
    DecisionCreate,
    DecisionRead,
    LegalSourceCreate,
    LegalSourceRead,
    RuleCreate,
    RuleUpdate,
    RuleRead,
    RuleList,
)

__all__ = [
    "ConditionCreate",
    "ConditionRead",
    "DecisionCreate",
    "DecisionRead",
    "LegalSourceCreate",
    "LegalSourceRead",
    "RuleCreate",
    "RuleUpdate",
    "RuleRead",
    "RuleList",
]
