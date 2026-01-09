"""API routes for embedding rules.

Re-exports from backend.rule_embedding_service.app.services.routes for backwards compatibility during migration.
"""
from backend.rule_embedding_service.app.services.routes import router

__all__ = ["router"]
