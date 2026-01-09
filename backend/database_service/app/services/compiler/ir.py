"""
Intermediate Representation (IR) types for compiled rules.

These Pydantic models represent the compile-time output that enables:
- O(1) rule lookup via premise index
- Linear condition evaluation (no tree traversal)
- Jump-table style decision lookup

Extended with jurisdiction support for v4 multi-jurisdiction architecture.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.core.ontology.jurisdiction import (
    Jurisdiction,
    JurisdictionCode,
    EquivalenceRef,
)


class ObligationSpec(BaseModel):
    """An obligation triggered by a decision."""

    id: str
    description: str | None = None
    deadline: str | None = None


class CompiledCheck(BaseModel):
    """A single flattened condition check.

    Conditions are compiled to a linear sequence for efficient evaluation.
    Each check can specify jump targets for control flow.
    """

    index: int
    """Position in the check sequence."""

    field: str
    """The fact field to check."""

    op: Literal["eq", "ne", "in", "not_in", "gt", "lt", "gte", "lte", "exists"]
    """The comparison operator."""

    value: Any = None
    """The value to compare against."""

    value_set: set[str] | None = None
    """Pre-computed set for O(1) 'in' operator lookups."""

    on_true: int | None = None
    """Jump target index if check passes (None = continue to next)."""

    on_false: int | None = None
    """Jump target index if check fails (None = continue to next)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump(self, *args, **kwargs) -> dict:
        """Override to convert set to list for JSON serialization."""
        data = super().model_dump(*args, **kwargs)
        if data.get("value_set") is not None:
            data["value_set"] = list(data["value_set"])
        return data

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> "CompiledCheck":
        """Override to convert list back to set for value_set."""
        if isinstance(obj, dict) and obj.get("value_set") is not None:
            obj = dict(obj)
            obj["value_set"] = set(obj["value_set"])
        return super().model_validate(obj, **kwargs)


class DecisionEntry(BaseModel):
    """A single entry in the decision table.

    The decision table replaces tree traversal with direct lookup.
    Each entry specifies conditions that must be satisfied for this outcome.
    """

    entry_id: int
    """Unique identifier within the decision table."""

    condition_mask: list[int]
    """
    Condition evaluation requirements:
    - Positive index (+i): check at index i must be True
    - Negative index (-i): check at index i must be False
    - Zero (0): don't care / wildcard
    """

    result: str
    """The decision result (e.g., 'authorized', 'not_authorized', 'exempt')."""

    obligations: list[ObligationSpec] = Field(default_factory=list)
    """Obligations triggered by this decision."""

    source_ref: str | None = None
    """Reference to source legal text (e.g., 'Article 36(2)')."""

    notes: str | None = None
    """Additional notes about this decision path."""


class RuleIR(BaseModel):
    """Complete Intermediate Representation of a compiled rule.

    This is the serialized format stored in the database and loaded at runtime.
    Extended with jurisdiction support for v4 architecture.
    """

    rule_id: str
    """Unique rule identifier."""

    version: int = 1
    """Rule content version."""

    ir_version: int = 2
    """IR format version for compatibility. v2 adds jurisdiction support."""

    # =========================================================================
    # Jurisdiction Scoping (v4 multi-jurisdiction support)
    # =========================================================================

    jurisdiction: Jurisdiction | None = None
    """Jurisdiction for this rule."""

    jurisdiction_code: JurisdictionCode = JurisdictionCode.EU
    """Jurisdiction code for quick filtering."""

    regime_id: str = "mica_2023"
    """Regulatory regime identifier."""

    cross_border_relevant: bool = False
    """Whether this rule applies in cross-border scenarios."""

    equivalence_refs: list[EquivalenceRef] = Field(default_factory=list)
    """Cross-border equivalence references."""

    conflicts_with: list[str] = Field(default_factory=list)
    """Rule IDs that may conflict with this rule."""

    # =========================================================================
    # O(1) Lookup Keys
    # =========================================================================

    premise_keys: list[str] = Field(default_factory=list)
    """
    Premise keys for inverted index lookup.
    Format: "field:value" (e.g., "instrument_type:art", "jurisdiction:EU")
    Now includes jurisdiction keys for O(1) filtered lookup.
    """

    # =========================================================================
    # Applicability Checks
    # =========================================================================

    applicability_checks: list[CompiledCheck] = Field(default_factory=list)
    """Flattened sequence of applicability checks."""

    applicability_mode: Literal["all", "any"] = "all"
    """How to combine applicability checks (AND vs OR)."""

    # =========================================================================
    # Decision Table
    # =========================================================================

    decision_checks: list[CompiledCheck] = Field(default_factory=list)
    """Flattened sequence of decision tree checks."""

    decision_table: list[DecisionEntry] = Field(default_factory=list)
    """Decision table entries for result lookup."""

    # =========================================================================
    # Dependency Graph (for multi-rule chaining)
    # =========================================================================

    produces: list[str] = Field(default_factory=list)
    """Facts this rule derives."""

    depends_on: list[str] = Field(default_factory=list)
    """Facts needed from other rules."""

    priority: int = 0
    """Topological order for rule chaining."""

    # =========================================================================
    # Pre-extracted Obligations
    # =========================================================================

    all_obligations: list[dict] = Field(default_factory=list)
    """All obligations extracted from decision tree for fast access."""

    # =========================================================================
    # Metadata
    # =========================================================================

    compiled_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    """ISO timestamp of when the rule was compiled."""

    source_hash: str | None = None
    """Hash of the source YAML for change detection."""

    source_document_id: str | None = None
    """Reference to source legal document."""

    source_article: str | None = None
    """Reference to source article."""

    def to_json(self) -> str:
        """Serialize to JSON string for database storage."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "RuleIR":
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)


# Type aliases for clarity
ConditionMask = list[int]
PremiseKey = str
