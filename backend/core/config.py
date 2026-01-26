"""Application configuration and feature flags."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # API
    app_name: str = "Regulatory KE Workbench"
    debug: bool = False

    # Optional ML features
    enable_vector_search: bool = False
    openai_api_key: str | None = None

    # Paths
    rules_dir: str = "backend/rules/data"
    data_dir: str = "data"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def ml_available() -> bool:
    """Check if ML dependencies are installed."""
    try:
        import sentence_transformers  # noqa: F401
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False
