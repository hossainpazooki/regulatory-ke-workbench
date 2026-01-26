"""KE domain - knowledge engineering workbench."""

# Re-export router from existing location for backwards compatibility
from backend.core.api.routes_ke import router

__all__ = ["router"]
