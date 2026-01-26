"""
Rule version repository for temporal versioning operations.

Provides CRUD operations for immutable rule version snapshots.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml
from sqlalchemy import text

from backend.storage.database import get_db
from backend.core.models import (
    RuleVersionRecord,
    generate_uuid,
    now_iso,
)


class RuleVersionRepository:
    """Repository for rule version persistence operations."""

    # =========================================================================
    # Version CRUD
    # =========================================================================

    def create_version(
        self,
        rule_id: str,
        content_yaml: str,
        effective_from: str | None = None,
        effective_to: str | None = None,
        created_by: str | None = None,
        jurisdiction_code: str | None = None,
        regime_id: str | None = None,
    ) -> RuleVersionRecord:
        """Create a new rule version.

        Automatically determines the version number and computes content hash.

        Args:
            rule_id: Unique rule identifier
            content_yaml: Original YAML content
            effective_from: Date when version becomes effective
            effective_to: Date when version expires
            created_by: Actor who created the version
            jurisdiction_code: Jurisdiction code (EU, UK, etc.)
            regime_id: Regulatory regime ID

        Returns:
            The created RuleVersionRecord
        """
        # Compute content hash
        content_hash = hashlib.sha256(content_yaml.encode()).hexdigest()[:16]

        # Parse YAML to JSON
        try:
            parsed = yaml.safe_load(content_yaml)
            content_json = json.dumps(parsed)
        except yaml.YAMLError:
            content_json = None

        with get_db() as conn:
            # Get next version number
            result = conn.execute(
                text("SELECT MAX(version) as max_version FROM rule_versions WHERE rule_id = :rule_id"),
                {"rule_id": rule_id},
            )
            row = result.fetchone()
            next_version = (row[0] or 0) + 1

            # Mark previous version as superseded
            if next_version > 1:
                conn.execute(
                    text("""
                    UPDATE rule_versions
                    SET superseded_by = :superseded_by, superseded_at = :superseded_at
                    WHERE rule_id = :rule_id AND version = :version
                    """),
                    {
                        "superseded_by": next_version,
                        "superseded_at": now_iso(),
                        "rule_id": rule_id,
                        "version": next_version - 1,
                    },
                )

            # Create new version record
            record = RuleVersionRecord(
                rule_id=rule_id,
                version=next_version,
                content_yaml=content_yaml,
                content_json=content_json,
                content_hash=content_hash,
                effective_from=effective_from,
                effective_to=effective_to,
                created_by=created_by,
                jurisdiction_code=jurisdiction_code,
                regime_id=regime_id,
            )

            conn.execute(
                text("""
                INSERT INTO rule_versions (
                    id, rule_id, version, content_yaml, content_json, content_hash,
                    effective_from, effective_to, created_at, created_by,
                    jurisdiction_code, regime_id
                ) VALUES (:id, :rule_id, :version, :content_yaml, :content_json, :content_hash,
                          :effective_from, :effective_to, :created_at, :created_by,
                          :jurisdiction_code, :regime_id)
                """),
                {
                    "id": record.id,
                    "rule_id": record.rule_id,
                    "version": record.version,
                    "content_yaml": record.content_yaml,
                    "content_json": record.content_json,
                    "content_hash": record.content_hash,
                    "effective_from": record.effective_from,
                    "effective_to": record.effective_to,
                    "created_at": record.created_at,
                    "created_by": record.created_by,
                    "jurisdiction_code": record.jurisdiction_code,
                    "regime_id": record.regime_id,
                },
            )

            conn.commit()
            return record

    def get_version(self, rule_id: str, version: int) -> RuleVersionRecord | None:
        """Get a specific version of a rule.

        Args:
            rule_id: The rule identifier
            version: The version number

        Returns:
            RuleVersionRecord if found, None otherwise
        """
        with get_db() as conn:
            result = conn.execute(
                text("SELECT * FROM rule_versions WHERE rule_id = :rule_id AND version = :version"),
                {"rule_id": rule_id, "version": version},
            )
            row = result.fetchone()

            if row:
                return RuleVersionRecord.from_row(row._mapping)
            return None

    def get_latest_version(self, rule_id: str) -> RuleVersionRecord | None:
        """Get the latest version of a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            RuleVersionRecord if found, None otherwise
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT * FROM rule_versions
                WHERE rule_id = :rule_id
                ORDER BY version DESC
                LIMIT 1
                """),
                {"rule_id": rule_id},
            )
            row = result.fetchone()

            if row:
                return RuleVersionRecord.from_row(row._mapping)
            return None

    def get_version_at_timestamp(
        self, rule_id: str, timestamp: str
    ) -> RuleVersionRecord | None:
        """Get the version effective at a specific timestamp.

        Uses effective_from and effective_to dates to find the correct version.
        Falls back to created_at if effective dates are not set.

        Args:
            rule_id: The rule identifier
            timestamp: ISO 8601 timestamp

        Returns:
            RuleVersionRecord if found, None otherwise
        """
        with get_db() as conn:
            # Try effective dates first
            result = conn.execute(
                text("""
                SELECT * FROM rule_versions
                WHERE rule_id = :rule_id
                  AND (effective_from IS NULL OR effective_from <= :ts)
                  AND (effective_to IS NULL OR effective_to > :ts)
                ORDER BY version DESC
                LIMIT 1
                """),
                {"rule_id": rule_id, "ts": timestamp},
            )
            row = result.fetchone()

            if row:
                return RuleVersionRecord.from_row(row._mapping)

            # Fall back to created_at
            result = conn.execute(
                text("""
                SELECT * FROM rule_versions
                WHERE rule_id = :rule_id AND created_at <= :ts
                ORDER BY version DESC
                LIMIT 1
                """),
                {"rule_id": rule_id, "ts": timestamp},
            )
            row = result.fetchone()

            if row:
                return RuleVersionRecord.from_row(row._mapping)
            return None

    def get_version_history(
        self, rule_id: str, limit: int = 100
    ) -> list[RuleVersionRecord]:
        """Get the version history for a rule.

        Args:
            rule_id: The rule identifier
            limit: Maximum number of versions to return

        Returns:
            List of RuleVersionRecord objects, newest first
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT * FROM rule_versions
                WHERE rule_id = :rule_id
                ORDER BY version DESC
                LIMIT :limit
                """),
                {"rule_id": rule_id, "limit": limit},
            )
            return [RuleVersionRecord.from_row(row._mapping) for row in result.fetchall()]

    def get_versions_by_hash(self, content_hash: str) -> list[RuleVersionRecord]:
        """Get all versions with a specific content hash.

        Useful for detecting duplicate content across rules.

        Args:
            content_hash: The content hash to search for

        Returns:
            List of RuleVersionRecord objects
        """
        with get_db() as conn:
            result = conn.execute(
                text("SELECT * FROM rule_versions WHERE content_hash = :content_hash ORDER BY rule_id, version"),
                {"content_hash": content_hash},
            )
            return [RuleVersionRecord.from_row(row._mapping) for row in result.fetchall()]

    def get_all_rule_ids(self) -> list[str]:
        """Get all unique rule IDs with versions.

        Returns:
            List of rule_id strings
        """
        with get_db() as conn:
            result = conn.execute(
                text("SELECT DISTINCT rule_id FROM rule_versions ORDER BY rule_id")
            )
            return [row[0] for row in result.fetchall()]

    def count_versions(self, rule_id: str | None = None) -> int:
        """Count versions.

        Args:
            rule_id: If provided, count only for this rule

        Returns:
            Number of versions
        """
        with get_db() as conn:
            if rule_id:
                result = conn.execute(
                    text("SELECT COUNT(*) as count FROM rule_versions WHERE rule_id = :rule_id"),
                    {"rule_id": rule_id},
                )
            else:
                result = conn.execute(text("SELECT COUNT(*) as count FROM rule_versions"))

            return result.fetchone()[0]
