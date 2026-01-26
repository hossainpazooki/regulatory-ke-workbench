"""
Verification repository for database operations.

Provides CRUD operations for verification results, evidence, and human reviews.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from backend.storage.database import get_db
from backend.core.models import (
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
            result = conn.execute(
                text("SELECT id FROM verification_results WHERE rule_id = :rule_id"),
                {"rule_id": rule_id}
            )
            existing = result.fetchone()

            result_id = existing[0] if existing else generate_uuid()

            if existing:
                # Update existing
                conn.execute(
                    text("""
                    UPDATE verification_results SET
                        status = :status,
                        confidence = :confidence,
                        verified_at = :verified_at,
                        verified_by = :verified_by,
                        notes = :notes
                    WHERE id = :id
                    """),
                    {
                        "status": status,
                        "confidence": confidence,
                        "verified_at": now_iso(),
                        "verified_by": verified_by,
                        "notes": notes,
                        "id": result_id,
                    },
                )

                # Delete old evidence
                conn.execute(
                    text("DELETE FROM verification_evidence WHERE verification_id = :verification_id"),
                    {"verification_id": result_id},
                )
            else:
                # Create new
                conn.execute(
                    text("""
                    INSERT INTO verification_results (
                        id, rule_id, rule_version, status, confidence,
                        verified_at, verified_by, notes
                    ) VALUES (:id, :rule_id, :rule_version, :status, :confidence,
                              :verified_at, :verified_by, :notes)
                    """),
                    {
                        "id": result_id,
                        "rule_id": rule_id,
                        "rule_version": 1,  # Default version
                        "status": status,
                        "confidence": confidence,
                        "verified_at": now_iso(),
                        "verified_by": verified_by,
                        "notes": notes,
                    },
                )

            # Insert evidence
            if evidence:
                for ev in evidence:
                    evidence_id = generate_uuid()
                    conn.execute(
                        text("""
                        INSERT INTO verification_evidence (
                            id, verification_id, tier, category, label,
                            score, details, source_span, rule_element, created_at
                        ) VALUES (:id, :verification_id, :tier, :category, :label,
                                  :score, :details, :source_span, :rule_element, :created_at)
                        """),
                        {
                            "id": evidence_id,
                            "verification_id": result_id,
                            "tier": ev.get("tier", 0),
                            "category": ev.get("category", "unknown"),
                            "label": ev.get("label", "warning"),
                            "score": ev.get("score"),
                            "details": ev.get("details"),
                            "source_span": ev.get("source_span"),
                            "rule_element": ev.get("rule_element"),
                            "created_at": now_iso(),
                        },
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
            result = conn.execute(
                text("SELECT * FROM verification_results WHERE rule_id = :rule_id"),
                {"rule_id": rule_id}
            )
            row = result.fetchone()

            if not row:
                return None, []

            record = VerificationResultRecord.from_row(row._mapping)

            # Get evidence
            result = conn.execute(
                text("""
                SELECT * FROM verification_evidence
                WHERE verification_id = :verification_id
                ORDER BY tier, category
                """),
                {"verification_id": record.id},
            )
            evidence = [
                VerificationEvidenceRecord.from_row(ev._mapping)
                for ev in result.fetchall()
            ]

            return record, evidence

    def get_all_verification_results(
        self,
    ) -> dict[str, tuple[VerificationResultRecord, list[VerificationEvidenceRecord]]]:
        """Get all verification results with evidence.

        Returns:
            Dict mapping rule_id to (result, evidence) tuples
        """
        with get_db() as conn:
            # Get all results
            result = conn.execute(
                text("SELECT * FROM verification_results ORDER BY rule_id")
            )
            results = {}
            for row in result.fetchall():
                results[row._mapping["rule_id"]] = VerificationResultRecord.from_row(row._mapping)

            # Get all evidence
            result = conn.execute(
                text("""
                SELECT ve.*, vr.rule_id
                FROM verification_evidence ve
                JOIN verification_results vr ON ve.verification_id = vr.id
                ORDER BY vr.rule_id, ve.tier, ve.category
                """)
            )

            evidence_by_rule: dict[str, list[VerificationEvidenceRecord]] = {}
            for row in result.fetchall():
                rule_id = row._mapping["rule_id"]
                if rule_id not in evidence_by_rule:
                    evidence_by_rule[rule_id] = []
                evidence_by_rule[rule_id].append(
                    VerificationEvidenceRecord.from_row(row._mapping)
                )

            return {
                rule_id: (rec, evidence_by_rule.get(rule_id, []))
                for rule_id, rec in results.items()
            }

    def delete_verification_result(self, rule_id: str) -> bool:
        """Delete verification result for a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            True if a result was deleted
        """
        with get_db() as conn:
            result = conn.execute(
                text("DELETE FROM verification_results WHERE rule_id = :rule_id"),
                {"rule_id": rule_id}
            )
            conn.commit()
            return result.rowcount > 0

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_verification_stats(self) -> dict[str, int]:
        """Get verification statistics by status.

        Returns:
            Dict with counts per status
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT status, COUNT(*) as count
                FROM verification_results
                GROUP BY status
                """)
            )
            return {row[0]: row[1] for row in result.fetchall()}

    def get_evidence_stats(self) -> dict[str, dict[str, int]]:
        """Get evidence statistics by tier and label.

        Returns:
            Dict mapping tier to {label: count}
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT tier, label, COUNT(*) as count
                FROM verification_evidence
                GROUP BY tier, label
                """)
            )

            stats: dict[str, dict[str, int]] = {}
            for row in result.fetchall():
                tier_key = f"tier_{row[0]}"
                if tier_key not in stats:
                    stats[tier_key] = {}
                stats[tier_key][row[1]] = row[2]

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
        record = ReviewRecord(
            rule_id=rule_id,
            reviewer_id=reviewer_id,
            decision=decision,
            notes=notes,
            metadata=metadata,
        )

        with get_db() as conn:
            conn.execute(
                text("""
                INSERT INTO reviews (
                    id, rule_id, reviewer_id, decision, notes, created_at, metadata
                ) VALUES (:id, :rule_id, :reviewer_id, :decision, :notes, :created_at, :metadata)
                """),
                {
                    "id": record.id,
                    "rule_id": record.rule_id,
                    "reviewer_id": record.reviewer_id,
                    "decision": record.decision,
                    "notes": record.notes,
                    "created_at": record.created_at,
                    "metadata": json.dumps(record.metadata) if record.metadata else None,
                },
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
            result = conn.execute(
                text("""
                SELECT * FROM reviews
                WHERE rule_id = :rule_id
                ORDER BY created_at DESC
                """),
                {"rule_id": rule_id},
            )
            return [ReviewRecord.from_row(row._mapping) for row in result.fetchall()]

    def get_latest_review(self, rule_id: str) -> ReviewRecord | None:
        """Get the most recent review for a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            Most recent ReviewRecord or None
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT * FROM reviews
                WHERE rule_id = :rule_id
                ORDER BY created_at DESC
                LIMIT 1
                """),
                {"rule_id": rule_id},
            )
            row = result.fetchone()

            if row:
                return ReviewRecord.from_row(row._mapping)
            return None

    def get_reviews_by_reviewer(self, reviewer_id: str) -> list[ReviewRecord]:
        """Get all reviews by a specific reviewer.

        Args:
            reviewer_id: The reviewer identifier

        Returns:
            List of ReviewRecord objects
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT * FROM reviews
                WHERE reviewer_id = :reviewer_id
                ORDER BY created_at DESC
                """),
                {"reviewer_id": reviewer_id},
            )
            return [ReviewRecord.from_row(row._mapping) for row in result.fetchall()]

    def count_reviews(self) -> int:
        """Count total reviews.

        Returns:
            Number of reviews
        """
        with get_db() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM reviews"))
            return result.fetchone()[0]

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def clear_all_verifications(self) -> int:
        """Delete all verification results and evidence.

        Returns:
            Number of results deleted
        """
        with get_db() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM verification_results"))
            count = result.fetchone()[0]

            conn.execute(text("DELETE FROM verification_evidence"))
            conn.execute(text("DELETE FROM verification_results"))
            conn.commit()

            return count
