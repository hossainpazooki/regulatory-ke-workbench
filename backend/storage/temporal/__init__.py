"""
Temporal Engine package for KE Workbench.

Provides versioning and event sourcing for rule lifecycle management.
"""

from backend.storage.temporal.version_repo import (
    RuleVersionRepository,
)
from backend.storage.temporal.event_repo import (
    RuleEventRepository,
)
from backend.core.models import (
    RuleVersionRecord,
    RuleEventRecord,
    RuleEventType,
)

__all__ = [
    # Repositories
    "RuleVersionRepository",
    "RuleEventRepository",
    # Models (re-exported for convenience)
    "RuleVersionRecord",
    "RuleEventRecord",
    "RuleEventType",
]
