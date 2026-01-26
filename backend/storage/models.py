"""
Data models for persistence layer.

Re-exports from backend.core.models for backwards compatibility.
"""

from backend.core.models import (
    generate_uuid,
    now_iso,
    RuleRecord,
    VerificationResultRecord,
    VerificationEvidenceRecord,
    ReviewRecord,
    PremiseIndexRecord,
)

__all__ = [
    "generate_uuid",
    "now_iso",
    "RuleRecord",
    "VerificationResultRecord",
    "VerificationEvidenceRecord",
    "ReviewRecord",
    "PremiseIndexRecord",
]
