"""
Data models for persistence layer.

Uses dataclasses for lightweight, serialization-friendly record types.
These mirror the database schema and can be converted to/from Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json
import uuid


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Get current time as ISO 8601 string."""
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Rule Record
# =============================================================================


@dataclass
class RuleRecord:
    """Database record for a rule."""

    rule_id: str
    content_yaml: str
    id: str = field(default_factory=generate_uuid)
    version: int = 1

    # Parsed content
    content_json: str | None = None

    # Compiled IR
    rule_ir: str | None = None
    ir_version: int = 1
    compiled_at: str | None = None

    # Source reference
    source_document_id: str | None = None
    source_article: str | None = None

    # Timestamps
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    # Status
    is_active: bool = True

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> RuleRecord:
        """Create from database row."""
        return cls(
            id=row["id"],
            rule_id=row["rule_id"],
            version=row["version"],
            content_yaml=row["content_yaml"],
            content_json=row["content_json"],
            rule_ir=row["rule_ir"],
            ir_version=row["ir_version"] or 1,
            compiled_at=row["compiled_at"],
            source_document_id=row["source_document_id"],
            source_article=row["source_article"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=bool(row["is_active"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "version": self.version,
            "content_yaml": self.content_yaml,
            "content_json": self.content_json,
            "rule_ir": self.rule_ir,
            "ir_version": self.ir_version,
            "compiled_at": self.compiled_at,
            "source_document_id": self.source_document_id,
            "source_article": self.source_article,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": 1 if self.is_active else 0,
        }


# =============================================================================
# Verification Result Record
# =============================================================================


@dataclass
class VerificationResultRecord:
    """Database record for verification results."""

    rule_id: str
    status: str  # verified, needs_review, inconsistent, unverified
    id: str = field(default_factory=generate_uuid)
    rule_version: int = 1
    confidence: float | None = None
    verified_at: str = field(default_factory=now_iso)
    verified_by: str | None = None
    notes: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> VerificationResultRecord:
        """Create from database row."""
        return cls(
            id=row["id"],
            rule_id=row["rule_id"],
            rule_version=row["rule_version"],
            status=row["status"],
            confidence=row["confidence"],
            verified_at=row["verified_at"],
            verified_by=row["verified_by"],
            notes=row["notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "status": self.status,
            "confidence": self.confidence,
            "verified_at": self.verified_at,
            "verified_by": self.verified_by,
            "notes": self.notes,
        }


# =============================================================================
# Verification Evidence Record
# =============================================================================


@dataclass
class VerificationEvidenceRecord:
    """Database record for verification evidence."""

    verification_id: str
    tier: int
    category: str
    label: str  # pass, fail, warning
    id: str = field(default_factory=generate_uuid)
    score: float | None = None
    details: str | None = None
    source_span: str | None = None
    rule_element: str | None = None
    created_at: str = field(default_factory=now_iso)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> VerificationEvidenceRecord:
        """Create from database row."""
        return cls(
            id=row["id"],
            verification_id=row["verification_id"],
            tier=row["tier"],
            category=row["category"],
            label=row["label"],
            score=row["score"],
            details=row["details"],
            source_span=row["source_span"],
            rule_element=row["rule_element"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "verification_id": self.verification_id,
            "tier": self.tier,
            "category": self.category,
            "label": self.label,
            "score": self.score,
            "details": self.details,
            "source_span": self.source_span,
            "rule_element": self.rule_element,
            "created_at": self.created_at,
        }


# =============================================================================
# Review Record
# =============================================================================


@dataclass
class ReviewRecord:
    """Database record for human reviews."""

    rule_id: str
    reviewer_id: str
    decision: str  # consistent, inconsistent, unknown
    id: str = field(default_factory=generate_uuid)
    notes: str | None = None
    created_at: str = field(default_factory=now_iso)
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ReviewRecord:
        """Create from database row."""
        metadata = None
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        return cls(
            id=row["id"],
            rule_id=row["rule_id"],
            reviewer_id=row["reviewer_id"],
            decision=row["decision"],
            notes=row["notes"],
            created_at=row["created_at"],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "reviewer_id": self.reviewer_id,
            "decision": self.decision,
            "notes": self.notes,
            "created_at": self.created_at,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }


# =============================================================================
# Premise Index Record
# =============================================================================


@dataclass
class PremiseIndexRecord:
    """Database record for premise index entries."""

    premise_key: str  # e.g., "instrument_type:art"
    rule_id: str
    rule_version: int = 1
    premise_position: int | None = None
    selectivity: float = 0.5

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PremiseIndexRecord:
        """Create from database row."""
        return cls(
            premise_key=row["premise_key"],
            rule_id=row["rule_id"],
            rule_version=row["rule_version"],
            premise_position=row["premise_position"],
            selectivity=row["selectivity"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "premise_key": self.premise_key,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "premise_position": self.premise_position,
            "selectivity": self.selectivity,
        }
