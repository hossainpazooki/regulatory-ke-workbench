"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.storage import init_db

# Import routers from new domain structure
from backend.rag import router as rag_router
from backend.rules import decide_router, rules_router
from backend.ke import router as ke_router
from backend.production import router as production_router
from backend.jurisdiction import router as jurisdiction_router
from backend.decoder import router as decoder_router
from backend.analytics import router as analytics_router
from backend.market_risk import router as market_risk_router
from backend.embeddings import embedding_router

# New domain routers (previously unexposed functionality)
from backend.defi_risk import router as defi_risk_router
from backend.token_compliance import router as token_compliance_router
from backend.protocol_risk import router as protocol_risk_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    print(f"Starting {settings.app_name}...")
    print(f"Rules directory: {settings.rules_dir}")
    print(f"Vector search enabled: {settings.enable_vector_search}")

    # Initialize database
    print("Initializing database...")
    init_db()

    yield

    # Shutdown
    print("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Computational law platform for tokenized real-world assets (RWAs)",
        version="0.2.0",  # Version bump for flat domain restructure
        lifespan=lifespan,
    )

    # CORS middleware - configurable via CORS_ORIGINS env var
    cors_origins_str = os.getenv("CORS_ORIGINS", "*")
    cors_origins = cors_origins_str.split(",") if cors_origins_str != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include domain routers (existing endpoints - unchanged URLs)
    app.include_router(rag_router)           # /qa
    app.include_router(decide_router)        # /decide
    app.include_router(rules_router)         # /rules
    app.include_router(ke_router)            # /ke
    app.include_router(production_router)    # /v2
    app.include_router(jurisdiction_router)  # /navigate
    app.include_router(decoder_router)       # /decoder, /decoder/counterfactual
    app.include_router(analytics_router)     # /analytics
    app.include_router(market_risk_router)   # /risk
    app.include_router(embedding_router)     # /embedding/rules

    # NEW domain routers (new endpoints)
    app.include_router(defi_risk_router)         # /defi-risk
    app.include_router(token_compliance_router)  # /token-compliance
    app.include_router(protocol_risk_router)     # /protocol-risk

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": settings.app_name,
            "version": "0.2.0",
            "endpoints": {
                # Existing endpoints
                "qa": "/qa/ask - Factual Q&A",
                "decide": "/decide - Regulatory decisions",
                "rules": "/rules - Rule inspection",
                "ke": "/ke/* - Knowledge Engineering workbench",
                "v2": "/v2/* - Production API with compiled IR",
                "navigate": "/navigate - Cross-border compliance navigation",
                "decoder": "/decoder/* - Tiered explanation decoder and what-if analysis",
                "analytics": "/analytics/* - Rule comparison, clustering, similarity search",
                "risk": "/risk/* - Market risk analytics (VaR, CVaR, liquidity)",
                "embedding": "/embedding/rules - Embedding rule CRUD",
                # New endpoints
                "defi-risk": "/defi-risk/* - DeFi protocol risk scoring",
                "token-compliance": "/token-compliance/* - Howey test, GENIUS Act analysis",
                "protocol-risk": "/protocol-risk/* - Blockchain protocol risk assessment",
            },
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
