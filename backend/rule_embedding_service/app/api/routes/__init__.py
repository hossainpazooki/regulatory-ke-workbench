"""Rule embedding service API routes."""

from .embedding import router as embedding_router

__all__ = ["embedding_router"]
