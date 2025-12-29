"""
Verification repository for database operations.

Provides CRUD operations for verification results, evidence, and human reviews.
"""

from __future__ import annotations

from typing import Any

from backend.persistence.database import get_db
from backend.persistence.models import (
    VerificationResultRecord,
    VerificationEvidenceRecord,
    ReviewRecord,
    generate_uuid,
    now_iso,
)


class VerificationRepository:
    """Repository for verification persistence operations."""

    # =========================================================================
    # Verification Results
    # =========================================================================

    def save_verification_result(
        self,
        rule_id: str,
        status: str,
        confidence: float | None = None,
        verified_by: str | None = None,
        notes: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
    ) -> VerificationResultRecord:
        """Save a verification result with optional evidence.

        If a verification exists for the rule, updates it. Otherwise, creates new.

        Args:
            rule_id: The rule identifier
            status: Verification status (verified, needs_review, inconsistent, unverified)
            confidence: Confidence score 0.0 to 1.0
            verified_by: Who performed verification ('system' or 'human:username')
            notes: Optional notes
            evidence: List of evidence dicts with tier, category, label, score, details

        Returns:
            The saved VerificationResultRecord
        """
        with get_db() as conn:
            # Check if result exists for this rule
            cursor = conn.execute(
                "SELECT id FROM verification_results WHERE rule_id = ?", (rule_id,)
            )
            existing = cursor.fetchone()

            result_id = existing["id"] if existing else generate_uuid()

            if existing:
                # Update existing
                conn.execute(
                    """
                    UPDATE verification_results SET
                        status = ?,
                        confidence = ?,
                        verified_at = ?,
                        verified_by = ?,
                        notes = ?
                    WHERE id = ?
                    """,
                    (status, confidence, now_iso(), verified_by, notes, result_id),
                )

                # Delete old evidence
                conn.execute(
                    "DELETE FROM verification_evidence WHERE verification_id = ?",
                    (result_id,),
                )
            else:
                # Create new
                conn.execute(
                    """
                    INSERT INTO verification_results (
                        id, rule_id, rule_version, status, confidence,
                        verified_at, verified_by, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result_id,
                        rule_id,
                        1,  # Default version
                        status,
                        confidence,
                        now_iso(),
                        verified_by,
                        notes,
                    ),
                )

            # Insert evidence
            if evidence:
                for ev in evidence:
                    evidence_id = generate_uuid()
                    conn.execute(
                        """
                        INSERT INTO verification_evidence (
                            id, verification_id, tier, category, label,
                            score, details, source_span, rule_element, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            evidence_id,
                            result_id,
                            ev.get("tier", 0),
                            ev.get("category", "unknown"),
                            ev.get("label", "warning"),
                            ev.get("score"),
                            ev.get("details"),
                            ev.get("source_span"),
                            ev.get("rule_element"),
                            now_iso(),
                        ),
                    )

            conn.commit()

            return VerificationResultRecord(
                id=result_id,
                rule_id=rule_id,
                status=status,
                confidence=confidence,
                verified_by=verified_by,
                notes=notes,
            )

    def get_verification_result(
        self, rule_id: str
    ) -> tuple[VerificationResultRecord | None, list[VerificationEvidenceRecord]]:
        """Get verification result and evidence for a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            Tuple of (VerificationResultRecord or None, list of VerificationEvidenceRecord)
        """
        with get_db() as conn:
            # Get result
            cursor = conn.execute(
                "SELECT * FROM verification_results WHERE rule_id = ?", (rule_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None, []

            result = VerificationResultRecord.from_row(dict(row))

            # Get evidence
            cursor = conn.execute(
                """
                SELECT * FROM verification_evidence
                WHERE verification_id = ?
                ORDER BY tier, category
                """,
                (result.id,),
            )
            evidence = [
                VerificationEvidenceRecord.from_row(dict(ev))
                for ev in cursor.fetchall()
            ]

            return result, evidence

    def get_all_verification_results(
        self,
    ) -> dict[str, tuple[VerificationResultRecord, list[VerificationEvidenceRecord]]]:
        """Get all verification results with evidence.

        Returns:
            Dict mapping rule_id to (result, evidence) tuples
        """
        with get_db() as conn:
            # Get all results
            cursor = conn.execute(
                "SELECT * FROM verification_results ORDER BY rule_id"
            )
            results = {
                row["rule_id"]: VerificationResultRecord.from_row(dict(row))
                for row in cursor.fetchall()
            }

            # Get all evidence
            cursor = conn.execute(
                """
                SELECT ve.*, vr.rule_id
                FROM verification_evidence ve
                JOIN verification_results vr ON ve.verification_id = vr.id
                ORDER BY vr.rule_id, ve.tier, ve.category
                """
            )

            evidence_by_rule: dict[str, list[VerificationEvidenceRecord]] = {}
            for row in cursor.fetchall():
                rule_id = row["rule_id"]
                if rule_id not in evidence_by_rule:
                    evidence_by_rule[rule_id] = []
                evidence_by_rule[rule_id].append(
                    VerificationEvidenceRecord.from_row(dict(row))
                )

            return {
                rule_id: (result, evidence_by_rule.get(rule_id, []))
                for rule_id, result in results.items()
            }

    def delete_verification_result(self, rule_id: str) -> bool:
        """Delete verification result for a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            True if a result was deleted
        """
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM verification_results WHERE rule_id = ?", (rule_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_verification_stats(self) -> dict[str, int]:
        """Get verification statistics by status.

        Returns:
            Dict with counts per status
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM verification_results
                GROUP BY status
                """
            )
            return {row["status"]: row["count"] for row in cursor.fetchall()}

    def get_evidence_stats(self) -> dict[str, dict[str, int]]:
        """Get evidence statistics by tier and label.

        Returns:
            Dict mapping tier to {label: count}
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT tier, label, COUNT(*) as count
                FROM verification_evidence
                GROUP BY tier, label
                """
            )

            stats: dict[str, dict[str, int]] = {}
            for row in cursor.fetchall():
                tier_key = f"tier_{row['tier']}"
                if tier_key not in stats:
                    stats[tier_key] = {}
                stats[tier_key][row["label"]] = row["count"]

            return stats

    # =========================================================================
    # Human Reviews
    # =========================================================================

    def save_review(
        self,
        rule_id: str,
        reviewer_id: str,
        decision: str,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReviewRecord:
        """Save a human review.

        Args:
            rule_id: The rule identifier
            reviewer_id: Identifier of the reviewer
            decision: Review decision (consistent, inconsistent, unknown)
            notes: Optional review notes
            metadata: Optional metadata dict

        Returns:
            The saved ReviewRecord
        """
        import json

        record = ReviewRecord(
            rule_id=rule_id,
            reviewer_id=reviewer_id,
            decision=decision,
            notes=notes,
            metadata=metadata,
        )

        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO reviews (
                    id, rule_id, reviewer_id, decision, notes, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.rule_id,
                    record.reviewer_id,
                    record.decision,
                    record.notes,
                    record.created_at,
                    json.dumps(record.metadata) if record.metadata else None,
                ),
            )
            conn.commit()

        return record

    def get_reviews_for_rule(self, rule_id: str) -> list[ReviewRecord]:
        """Get all reviews for a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            List of ReviewRecord objects, ordered by creation time
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM reviews
                WHERE rule_id = ?
                ORDER BY created_at DESC
                """,
                (rule_id,),
            )
            return [ReviewRecord.from_row(dict(row)) for row in cursor.fetchall()]

    def get_latest_review(self, rule_id: str) -> ReviewRecord | None:
        """Get the most recent review for a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            Most recent ReviewRecord or None
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM reviews
                WHERE rule_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (rule_id,),
            )
            row = cursor.fetchone()

            if row:
                return ReviewRecord.from_row(dict(row))
            return None

    def get_reviews_by_reviewer(self, reviewer_id: str) -> list[ReviewRecord]:
        """Get all reviews by a specific reviewer.

        Args:
            reviewer_id: The reviewer identifier

        Returns:
            List of ReviewRecord objects
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM reviews
                WHERE reviewer_id = ?
                ORDER BY created_at DESC
                """,
                (reviewer_id,),
            )
            return [ReviewRecord.from_row(dict(row)) for row in cursor.fetchall()]

    def count_reviews(self) -> int:
        """Count total reviews.

        Returns:
            Number of reviews
        """
        with get_db() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM reviews")
            return cursor.fetchone()["count"]

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def clear_all_verifications(self) -> int:
        """Delete all verification results and evidence.

        Returns:
            Number of results deleted
        """
        with get_db() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM verification_results")
            count = cursor.fetchone()["count"]

            conn.execute("DELETE FROM verification_evidence")
            conn.execute("DELETE FROM verification_results")
            conn.commit()

            return count
