"""SQLModel models for embedding rules.

Re-exports from backend.rule_embedding_service.app.services.models for backwards compatibility during migration.
"""
from backend.rule_embedding_service.app.services.models import (
    EmbeddingRule,
    EmbeddingCondition,
    EmbeddingDecision,
    EmbeddingLegalSource,
)

__all__ = [
    "EmbeddingRule",
    "EmbeddingCondition",
    "EmbeddingDecision",
    "EmbeddingLegalSource",
]
