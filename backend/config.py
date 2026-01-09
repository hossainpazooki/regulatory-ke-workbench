"""DEPRECATED: Import from backend.core.config instead."""
# Re-export for backwards compatibility during migration
from backend.core.config import Settings, get_settings, ml_available

__all__ = ["Settings", "get_settings", "ml_available"]
