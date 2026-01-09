"""Routes for inspecting rules.

Re-exports from backend.core.api.routes_rules for backwards compatibility during migration.
"""
# Re-export router from existing location
from backend.core.api.routes_rules import router

__all__ = ["router"]
