"""Production domain - compiled IR and optimized rule evaluation."""

# Re-export router from existing location for backwards compatibility
from backend.core.api.routes_production import router

__all__ = ["router"]
