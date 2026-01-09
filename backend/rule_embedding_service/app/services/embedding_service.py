"""Business logic for embedding rules.

Re-exports from backend.rule_embedding_service.app.services.service for backwards compatibility during migration.
"""
from backend.rule_embedding_service.app.services.service import EmbeddingRuleService

__all__ = ["EmbeddingRuleService"]
