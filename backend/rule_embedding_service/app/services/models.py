"""SQLModel table definitions for embedding rules."""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


class EmbeddingRule(SQLModel, table=True):
    """Rule entity for embedding-based search."""

    __tablename__ = "embedding_rules"

    id: Optional[int] = Field(default=None, primary_key=True)
    rule_id: str = Field(..., unique=True, index=True, description="Human-readable rule ID")
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(default=None, description="Rule description")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )
    is_active: bool = Field(default=True, description="Whether the rule is active")

    conditions: list["EmbeddingCondition"] = Relationship(
        back_populates="rule",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"},
    )
    decision: Optional["EmbeddingDecision"] = Relationship(
        back_populates="rule",
        sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan", "lazy": "selectin"},
    )
    legal_sources: list["EmbeddingLegalSource"] = Relationship(
        back_populates="rule",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"},
    )


class EmbeddingCondition(SQLModel, table=True):
    """Rule condition for embedding rules."""

    __tablename__ = "embedding_conditions"

    id: Optional[int] = Field(default=None, primary_key=True)
    field: str = Field(..., description="Field path to check")
    operator: str = Field(..., description="Comparison operator")
    value: str = Field(..., description="JSON-serialized comparison value")
    description: Optional[str] = Field(default=None, description="Human-readable description")

    rule_id: int = Field(foreign_key="embedding_rules.id", ondelete="CASCADE")
    rule: Optional["EmbeddingRule"] = Relationship(back_populates="conditions")


class EmbeddingDecision(SQLModel, table=True):
    """Rule decision for embedding rules."""

    __tablename__ = "embedding_decisions"

    id: Optional[int] = Field(default=None, primary_key=True)
    outcome: str = Field(..., description="Decision outcome")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    explanation: Optional[str] = Field(default=None, description="Explanation")

    rule_id: int = Field(foreign_key="embedding_rules.id", unique=True, ondelete="CASCADE")
    rule: Optional["EmbeddingRule"] = Relationship(back_populates="decision")


class EmbeddingLegalSource(SQLModel, table=True):
    """Legal source for embedding rules."""

    __tablename__ = "embedding_legal_sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    citation: str = Field(..., description="Legal citation")
    document_id: Optional[str] = Field(default=None, description="Document identifier")
    url: Optional[str] = Field(default=None, description="URL to source")

    rule_id: int = Field(foreign_key="embedding_rules.id", ondelete="CASCADE")
    rule: Optional["EmbeddingRule"] = Relationship(back_populates="legal_sources")


EmbeddingRule.model_rebuild()
EmbeddingCondition.model_rebuild()
EmbeddingDecision.model_rebuild()
EmbeddingLegalSource.model_rebuild()
