"""API module - FastAPI routes."""

from .routes_qa import router as qa_router
from .routes_decide import router as decide_router
from .routes_rules import router as rules_router
from .routes_ke import router as ke_router
from .routes_production import router as production_router

__all__ = [
    "qa_router",
    "decide_router",
    "rules_router",
    "ke_router",
    "production_router",
]
