"""Rule service API package."""

from .routes.rules import router as rules_router
from .routes.decide import router as decide_router

__all__ = ["rules_router", "decide_router"]
