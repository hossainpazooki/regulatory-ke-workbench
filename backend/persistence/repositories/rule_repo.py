"""
Rule repository for database operations.

Provides CRUD operations for rules, including IR storage and premise index management.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from backend.persistence.database import get_db
from backend.persistence.models import (
    RuleRecord,
    PremiseIndexRecord,
    generate_uuid,
    now_iso,
)


class RuleRepository:
    """Repository for rule persistence operations."""

    # =========================================================================
    # Rule CRUD
    # =========================================================================

    def save_rule(
        self,
        rule_id: str,
        content_yaml: str,
        source_document_id: str | None = None,
        source_article: str | None = None,
    ) -> RuleRecord:
        """Save or update a rule.

        If the rule_id exists, updates the existing record.
        Otherwise, creates a new record.

        Args:
            rule_id: Unique rule identifier
            content_yaml: Original YAML content
            source_document_id: Reference to source legal document
            source_article: Reference to source article

        Returns:
            The saved RuleRecord
        """
        # Parse YAML to JSON for structured queries
        try:
            parsed = yaml.safe_load(content_yaml)
            content_json = json.dumps(parsed)
        except yaml.YAMLError:
            content_json = None

        with get_db() as conn:
            # Check if rule exists
            cursor = conn.execute(
                "SELECT id, version FROM rules WHERE rule_id = ?", (rule_id,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing rule
                record = RuleRecord(
                    id=existing["id"],
                    rule_id=rule_id,
                    version=existing["version"],
                    content_yaml=content_yaml,
                    content_json=content_json,
                    source_document_id=source_document_id,
                    source_article=source_article,
                    updated_at=now_iso(),
                )

                conn.execute(
                    """
                    UPDATE rules SET
                        content_yaml = ?,
                        content_json = ?,
                        source_document_id = ?,
                        source_article = ?,
                        updated_at = ?,
                        rule_ir = NULL,
                        compiled_at = NULL
                    WHERE id = ?
                    """,
                    (
                        content_yaml,
                        content_json,
                        source_document_id,
                        source_article,
                        record.updated_at,
                        record.id,
                    ),
                )
            else:
                # Create new rule
                record = RuleRecord(
                    rule_id=rule_id,
                    content_yaml=content_yaml,
                    content_json=content_json,
                    source_document_id=source_document_id,
                    source_article=source_article,
                )

                conn.execute(
                    """
                    INSERT INTO rules (
                        id, rule_id, version, content_yaml, content_json,
                        source_document_id, source_article, created_at, updated_at, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.rule_id,
                        record.version,
                        record.content_yaml,
                        record.content_json,
                        record.source_document_id,
                        record.source_article,
                        record.created_at,
                        record.updated_at,
                        1,
                    ),
                )

            conn.commit()
            return record

    def get_rule(self, rule_id: str) -> RuleRecord | None:
        """Get a rule by ID.

        Args:
            rule_id: The rule identifier

        Returns:
            RuleRecord if found, None otherwise
        """
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM rules WHERE rule_id = ? AND is_active = 1", (rule_id,)
            )
            row = cursor.fetchone()

            if row:
                return RuleRecord.from_row(dict(row))
            return None

    def get_all_rules(self, active_only: bool = True) -> list[RuleRecord]:
        """Get all rules.

        Args:
            active_only: If True, only return active rules

        Returns:
            List of RuleRecord objects
        """
        with get_db() as conn:
            if active_only:
                cursor = conn.execute(
                    "SELECT * FROM rules WHERE is_active = 1 ORDER BY rule_id"
                )
            else:
                cursor = conn.execute("SELECT * FROM rules ORDER BY rule_id")

            return [RuleRecord.from_row(dict(row)) for row in cursor.fetchall()]

    def delete_rule(self, rule_id: str, soft: bool = True) -> bool:
        """Delete a rule.

        Args:
            rule_id: The rule identifier
            soft: If True, mark as inactive instead of deleting

        Returns:
            True if a rule was deleted/deactivated
        """
        with get_db() as conn:
            if soft:
                cursor = conn.execute(
                    "UPDATE rules SET is_active = 0, updated_at = ? WHERE rule_id = ?",
                    (now_iso(), rule_id),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM rules WHERE rule_id = ?", (rule_id,)
                )

            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # IR (Intermediate Representation) Operations
    # =========================================================================

    def update_rule_ir(
        self,
        rule_id: str,
        rule_ir: str,
        ir_version: int = 1,
    ) -> bool:
        """Update the compiled IR for a rule.

        Args:
            rule_id: The rule identifier
            rule_ir: JSON string of the compiled IR
            ir_version: Version of the IR schema

        Returns:
            True if the rule was updated
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                UPDATE rules SET
                    rule_ir = ?,
                    ir_version = ?,
                    compiled_at = ?,
                    updated_at = ?
                WHERE rule_id = ? AND is_active = 1
                """,
                (rule_ir, ir_version, now_iso(), now_iso(), rule_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_rule_ir(self, rule_id: str) -> str | None:
        """Get the compiled IR for a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            JSON string of IR if compiled, None otherwise
        """
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT rule_ir FROM rules WHERE rule_id = ? AND is_active = 1",
                (rule_id,),
            )
            row = cursor.fetchone()

            if row:
                return row["rule_ir"]
            return None

    def get_rules_needing_compilation(self) -> list[str]:
        """Get rule IDs that need compilation (no IR or IR is outdated).

        Returns:
            List of rule_id strings
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT rule_id FROM rules
                WHERE is_active = 1 AND (rule_ir IS NULL OR compiled_at < updated_at)
                ORDER BY rule_id
                """
            )
            return [row["rule_id"] for row in cursor.fetchall()]

    # =========================================================================
    # Premise Index Operations
    # =========================================================================

    def update_premise_index(
        self,
        rule_id: str,
        premise_keys: list[str],
        rule_version: int = 1,
    ) -> None:
        """Update the premise index for a rule.

        Removes existing entries and adds new ones.

        Args:
            rule_id: The rule identifier
            premise_keys: List of premise keys (e.g., ["instrument_type:art"])
            rule_version: Rule version
        """
        with get_db() as conn:
            # Delete existing entries
            conn.execute(
                "DELETE FROM rule_premise_index WHERE rule_id = ? AND rule_version = ?",
                (rule_id, rule_version),
            )

            # Insert new entries
            for position, key in enumerate(premise_keys):
                conn.execute(
                    """
                    INSERT INTO rule_premise_index (
                        premise_key, rule_id, rule_version, premise_position, selectivity
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, rule_id, rule_version, position, 0.5),
                )

            conn.commit()

    def get_rules_by_premise(self, premise_key: str) -> list[str]:
        """Get rule IDs that match a premise key.

        Args:
            premise_key: The premise key to search for (e.g., "instrument_type:art")

        Returns:
            List of matching rule_id strings
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT rpi.rule_id
                FROM rule_premise_index rpi
                JOIN rules r ON rpi.rule_id = r.rule_id
                WHERE rpi.premise_key = ? AND r.is_active = 1
                ORDER BY rpi.rule_id
                """,
                (premise_key,),
            )
            return [row["rule_id"] for row in cursor.fetchall()]

    def get_rules_by_premises(self, premise_keys: list[str]) -> list[str]:
        """Get rule IDs that match ALL premise keys (intersection).

        Args:
            premise_keys: List of premise keys to match

        Returns:
            List of rule_id strings that match all keys
        """
        if not premise_keys:
            return []

        with get_db() as conn:
            # Find rules that have entries for all premise keys
            placeholders = ",".join(["?"] * len(premise_keys))
            cursor = conn.execute(
                f"""
                SELECT rpi.rule_id
                FROM rule_premise_index rpi
                JOIN rules r ON rpi.rule_id = r.rule_id
                WHERE rpi.premise_key IN ({placeholders}) AND r.is_active = 1
                GROUP BY rpi.rule_id
                HAVING COUNT(DISTINCT rpi.premise_key) = ?
                ORDER BY rpi.rule_id
                """,
                (*premise_keys, len(premise_keys)),
            )
            return [row["rule_id"] for row in cursor.fetchall()]

    def get_all_premise_keys(self) -> list[str]:
        """Get all unique premise keys in the index.

        Returns:
            List of unique premise key strings
        """
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT premise_key FROM rule_premise_index ORDER BY premise_key"
            )
            return [row["premise_key"] for row in cursor.fetchall()]

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_rules_by_document(self, document_id: str) -> list[RuleRecord]:
        """Get all rules from a specific document.

        Args:
            document_id: The source document identifier

        Returns:
            List of RuleRecord objects
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM rules
                WHERE source_document_id = ? AND is_active = 1
                ORDER BY source_article, rule_id
                """,
                (document_id,),
            )
            return [RuleRecord.from_row(dict(row)) for row in cursor.fetchall()]

    def count_rules(self, active_only: bool = True) -> int:
        """Count total rules.

        Args:
            active_only: If True, only count active rules

        Returns:
            Number of rules
        """
        with get_db() as conn:
            if active_only:
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM rules WHERE is_active = 1"
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as count FROM rules")

            return cursor.fetchone()["count"]

    def count_compiled_rules(self) -> int:
        """Count rules with compiled IR.

        Returns:
            Number of rules with IR
        """
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM rules WHERE is_active = 1 AND rule_ir IS NOT NULL"
            )
            return cursor.fetchone()["count"]
