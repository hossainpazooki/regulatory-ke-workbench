"""API module - FastAPI routes."""

from .routes_decide import router as decide_router
from .routes_rules import router as rules_router
from .routes_ke import router as ke_router
from .routes_production import router as production_router
from .routes_navigate import router as navigate_router
from .routes_risk import router as risk_router
# Domain routers (migrated)
from backend.analytics import router as analytics_router
from backend.decoder import router as decoder_router
from backend.embeddings import embedding_router

__all__ = [
    "decide_router",
    "rules_router",
    "ke_router",
    "production_router",
    "navigate_router",
    "decoder_router",
    "analytics_router",
    "risk_router",
    "embedding_router",
]
