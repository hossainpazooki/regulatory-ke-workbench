"""Pydantic models for the rule DSL schema.

These models mirror the OCaml rule_dsl.ml types and the YAML rule specification
in docs/rule_dsl.md. They include the consistency block defined in
docs/semantic_consistency_regulatory_kg.md.

Extended with jurisdiction support for v4 multi-jurisdiction architecture.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

from backend.core.ontology.jurisdiction import JurisdictionCode


# =============================================================================
# Source Reference
# =============================================================================

class SourceRef(BaseModel):
    """Source reference linking a rule to its legal text.

    Maps to OCaml: Rule_dsl.source
    """
    document_id: str = Field(..., description="Document identifier (e.g., 'mica_2023')")
    article: str | None = Field(None, description="Article number (e.g., '36(1)')")
    section: str | None = Field(None, description="Section identifier")
    paragraphs: list[str] = Field(default_factory=list, description="Paragraph references")
    pages: list[int] = Field(default_factory=list, description="Page numbers")
    url: str | None = Field(None, description="URL to source document")


# =============================================================================
# Condition Expressions
# =============================================================================

class ComparisonOp(str, Enum):
    """Comparison operators for conditions.

    Maps to OCaml: Rule_dsl.comparison_op
    """
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"


class ConditionSpec(BaseModel):
    """A single condition specification.

    Maps to OCaml: Rule_dsl.condition_expr (FieldCheck variant)
    """
    field: str = Field(..., description="Field name to evaluate")
    operator: str = Field("==", description="Comparison operator")
    value: Any = Field(None, description="Expected value")
    description: str | None = Field(None, description="Human-readable description")


class ConditionGroup(BaseModel):
    """Grouped conditions with logical operators.

    Maps to OCaml: Rule_dsl.condition_expr (AllOf/AnyOf variants)
    """
    all: list[ConditionSpec | ConditionGroup] | None = Field(
        None, description="All conditions must be true (AND)"
    )
    any: list[ConditionSpec | ConditionGroup] | None = Field(
        None, description="Any condition must be true (OR)"
    )


# =============================================================================
# Decision Tree
# =============================================================================

class ObligationSpec(BaseModel):
    """An obligation triggered by a decision.

    Maps to OCaml: part of Rule_dsl.decision_leaf
    """
    id: str = Field(..., description="Obligation identifier")
    description: str | None = Field(None, description="Human-readable description")
    deadline: str | None = Field(None, description="Deadline specification")
    source_ref: str | None = Field(None, description="Source article reference")


class DecisionLeaf(BaseModel):
    """A leaf node (terminal result) in the decision tree.

    Maps to OCaml: Rule_dsl.decision_node (Leaf variant)
    """
    result: str = Field(..., description="Decision outcome")
    obligations: list[ObligationSpec] = Field(default_factory=list)
    notes: str | None = Field(None, description="Explanation notes")


class DecisionBranch(BaseModel):
    """A branch node in the decision tree.

    Maps to OCaml: Rule_dsl.decision_node (Branch variant)
    """
    node_id: str = Field(..., description="Node identifier for tracing")
    condition: ConditionSpec | None = Field(None, description="Branch condition")
    true_branch: DecisionBranch | DecisionLeaf | None = Field(
        None, description="Path if condition is true"
    )
    false_branch: DecisionBranch | DecisionLeaf | None = Field(
        None, description="Path if condition is false"
    )


# =============================================================================
# Consistency / QA Metadata
# =============================================================================

class ConsistencyStatus(str, Enum):
    """Status of consistency verification."""
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    INCONSISTENT = "inconsistent"
    UNVERIFIED = "unverified"


class ConsistencyLabel(str, Enum):
    """Labels for consistency evidence."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ENTAILS = "entails"
    NEUTRAL = "neutral"
    CONTRADICTS = "contradicts"


class ConsistencyEvidence(BaseModel):
    """A single piece of consistency verification evidence.

    See docs/semantic_consistency_regulatory_kg.md for tier definitions.
    """
    tier: int = Field(..., ge=0, le=4, description="Verification tier (0-4)")
    category: str = Field(..., description="Check category (e.g., 'deontic_alignment')")
    label: str = Field(..., description="Result label (pass/fail/warning/etc.)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    details: str = Field(..., description="Human-readable explanation")
    source_span: str | None = Field(None, description="Relevant text from source")
    rule_element: str | None = Field(None, description="Path to rule element")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        description="ISO 8601 timestamp"
    )


class ConsistencySummary(BaseModel):
    """Summary of consistency verification results."""
    status: ConsistencyStatus = Field(
        default=ConsistencyStatus.UNVERIFIED,
        description="Overall verification status"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted confidence score"
    )
    last_verified: str | None = Field(None, description="ISO 8601 timestamp")
    verified_by: str | None = Field(None, description="'system' or 'human:username'")
    notes: str | None = Field(None, description="Optional reviewer notes")


class ConsistencyBlock(BaseModel):
    """Complete consistency metadata for a rule."""
    summary: ConsistencySummary = Field(default_factory=ConsistencySummary)
    evidence: list[ConsistencyEvidence] = Field(default_factory=list)


# =============================================================================
# Rule Metadata
# =============================================================================

class RuleMetadata(BaseModel):
    """Metadata about a rule.

    Maps to OCaml: Rule_dsl.rule_metadata
    """
    version: str = Field(default="1.0", description="Semantic version")
    author: str | None = Field(None, description="Rule author")
    reviewed_by: str | None = Field(None, description="Reviewer")
    last_updated: str | None = Field(None, description="Last update timestamp")
    tags: list[str] = Field(default_factory=list, description="Classification tags")


# =============================================================================
# Complete Rule
# =============================================================================

class Rule(BaseModel):
    """A complete rule specification.

    Maps to OCaml: Rule_dsl.rule
    Extended with jurisdiction support for v4 architecture.
    """
    # Identity
    rule_id: str = Field(..., description="Unique rule identifier")

    # Metadata
    version: str = Field(default="1.0", description="Rule version")
    description: str | None = Field(None, description="Human-readable description")
    effective_from: date | None = Field(None, description="When rule becomes active")
    effective_to: date | None = Field(None, description="When rule expires")
    tags: list[str] = Field(default_factory=list, description="Classification tags")

    # Jurisdiction scoping (v4 multi-jurisdiction support)
    jurisdiction: JurisdictionCode = Field(
        default=JurisdictionCode.EU,
        description="Primary jurisdiction for this rule"
    )
    regime_id: str = Field(
        default="mica_2023",
        description="Regulatory regime identifier (e.g., mica_2023, fca_crypto_2024)"
    )
    cross_border_relevant: bool = Field(
        default=False,
        description="Whether this rule applies in cross-border scenarios"
    )

    # Logic
    applies_if: ConditionGroup | None = Field(
        None, description="Applicability conditions"
    )
    decision_tree: DecisionBranch | DecisionLeaf | None = Field(
        None, description="Decision logic"
    )

    # Source
    source: SourceRef | None = Field(None, description="Source citation")
    interpretation_notes: str | None = Field(
        None, description="Explanation of modeling choices"
    )

    # QA / Consistency
    consistency: ConsistencyBlock | None = Field(
        None, description="Verification metadata"
    )

    # Extended metadata
    metadata: RuleMetadata | None = Field(None, description="Additional metadata")


# =============================================================================
# Rule Pack (Collection)
# =============================================================================

class RulePack(BaseModel):
    """A collection of related rules.

    Maps to OCaml: Rule_dsl.rule_pack
    """
    pack_id: str = Field(..., description="Pack identifier")
    name: str = Field(..., description="Pack name")
    description: str = Field(..., description="Pack description")
    version: str = Field(default="1.0", description="Pack version")
    rules: list[Rule] = Field(default_factory=list, description="Rules in this pack")


# Enable forward references for recursive types
ConditionGroup.model_rebuild()
DecisionBranch.model_rebuild()
