"""
Persistence layer for KE Workbench.

This module provides database persistence for rules, verification results,
and human reviews. Uses SQLite by default with PostgreSQL-compatible schema.
"""

from backend.persistence.database import (
    get_db,
    init_db,
    get_db_path,
    reset_db,
    get_table_stats,
)

from backend.persistence.models import (
    RuleRecord,
    VerificationResultRecord,
    VerificationEvidenceRecord,
    ReviewRecord,
    PremiseIndexRecord,
)

from backend.persistence.repositories.rule_repo import RuleRepository
from backend.persistence.repositories.verification_repo import VerificationRepository

from backend.persistence.migration import (
    migrate_yaml_rules,
    sync_rule_to_db,
    load_rules_from_db,
    extract_premise_keys,
    get_migration_status,
)

__all__ = [
    # Database
    "get_db",
    "init_db",
    "get_db_path",
    "reset_db",
    "get_table_stats",
    # Models
    "RuleRecord",
    "VerificationResultRecord",
    "VerificationEvidenceRecord",
    "ReviewRecord",
    "PremiseIndexRecord",
    # Repositories
    "RuleRepository",
    "VerificationRepository",
    # Migration
    "migrate_yaml_rules",
    "sync_rule_to_db",
    "load_rules_from_db",
    "extract_premise_keys",
    "get_migration_status",
]
