"""Rule embedding service API package."""

from .routes.embedding import router as embedding_router

__all__ = ["embedding_router"]
