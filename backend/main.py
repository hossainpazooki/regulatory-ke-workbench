"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.core.api import (
    qa_router,
    decide_router,
    rules_router,
    ke_router,
    production_router,
    navigate_router,
    decoder_router,
    counterfactual_router,
    analytics_router,
    embedding_router,
)
from backend.database_service.app.services import init_db


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
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(qa_router)
    app.include_router(decide_router)
    app.include_router(rules_router)
    app.include_router(ke_router)
    app.include_router(production_router)
    app.include_router(navigate_router)
    app.include_router(decoder_router)
    app.include_router(counterfactual_router)
    app.include_router(analytics_router)
    app.include_router(embedding_router)

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": settings.app_name,
            "version": "0.1.0",
            "endpoints": {
                "qa": "/qa/ask - Factual Q&A",
                "decide": "/decide - Regulatory decisions",
                "rules": "/rules - Rule inspection",
                "ke": "/ke/* - Knowledge Engineering workbench",
                "v2": "/v2/* - Production API with compiled IR",
                "navigate": "/navigate - Cross-border compliance navigation",
                "decoder": "/decoder/* - Tiered explanation decoder",
                "counterfactual": "/counterfactual/* - What-if analysis",
                "analytics": "/analytics/* - Rule comparison, clustering, similarity search",
                "embedding": "/embedding/rules - Embedding rule CRUD",
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
