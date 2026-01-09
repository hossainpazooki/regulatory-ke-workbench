"""Routes for regulatory decisions.

Re-exports from backend.core.api.routes_decide for backwards compatibility during migration.
"""
# Re-export router from existing location
from backend.core.api.routes_decide import router

__all__ = ["router"]
