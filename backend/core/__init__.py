"""Core package - Shared configuration, database, and domain types."""

from .config import Settings, get_settings, ml_available
from .database import get_engine, get_session, init_sqlmodel_tables
from .models import (
    generate_uuid,
    now_iso,
    RuleRecord,
    VerificationResultRecord,
    VerificationEvidenceRecord,
    ReviewRecord,
    PremiseIndexRecord,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "ml_available",
    # Database
    "get_engine",
    "get_session",
    "init_sqlmodel_tables",
    # Models
    "generate_uuid",
    "now_iso",
    "RuleRecord",
    "VerificationResultRecord",
    "VerificationEvidenceRecord",
    "ReviewRecord",
    "PremiseIndexRecord",
]
