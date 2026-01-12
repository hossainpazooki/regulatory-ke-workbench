"""API module - FastAPI routes."""

from .routes_qa import router as qa_router
from .routes_decide import router as decide_router
from .routes_rules import router as rules_router
from .routes_ke import router as ke_router
from .routes_production import router as production_router
from .routes_navigate import router as navigate_router
from .routes_decoder import router as decoder_router
from .routes_counterfactual import router as counterfactual_router
from .routes_analytics import router as analytics_router
from backend.rule_embedding_service.app.services.routes import router as embedding_router

__all__ = [
    "qa_router",
    "decide_router",
    "rules_router",
    "ke_router",
    "production_router",
    "navigate_router",
    "decoder_router",
    "counterfactual_router",
    "analytics_router",
    "embedding_router",
]
