"""Pydantic models for API requests and responses."""

from typing import Any
from pydantic import BaseModel, Field


# =============================================================================
# Q&A Models
# =============================================================================


class AskRequest(BaseModel):
    """Request for factual Q&A."""

    question: str = Field(..., description="The question to answer")
    top_k: int = Field(5, description="Number of context chunks to retrieve")


class SourceCitation(BaseModel):
    """A source citation in a Q&A response."""

    document_id: str
    chunk_id: str
    section: str | None = None
    score: float


class AskResponse(BaseModel):
    """Response for factual Q&A."""

    answer: str
    sources: list[SourceCitation]
    method: str = Field(..., description="How the answer was generated: 'llm' or 'excerpt'")


# =============================================================================
# Decision Models
# =============================================================================


class DecideRequest(BaseModel):
    """Request for regulatory decision."""

    # Common fields
    instrument_type: str | None = None
    activity: str | None = None
    jurisdiction: str | None = None
    authorized: bool | None = None

    # Actor attributes
    actor_type: str | None = None
    issuer_type: str | None = None
    is_credit_institution: bool | None = None
    is_authorized_institution: bool | None = None

    # Instrument attributes
    reference_asset: str | None = None
    is_significant: bool | None = None
    reserve_value_eur: float | None = None

    # Additional fields (flexible)
    extra: dict[str, Any] = Field(default_factory=dict)

    # Options
    rule_id: str | None = Field(None, description="Evaluate specific rule only")


class TraceStepResponse(BaseModel):
    """A step in the decision trace."""

    node: str
    condition: str
    result: bool
    value_checked: Any = None


class ObligationResponse(BaseModel):
    """An obligation from a decision."""

    id: str
    description: str | None = None
    source: str | None = None
    deadline: str | None = None


class DecisionResponse(BaseModel):
    """Response for a single rule evaluation."""

    rule_id: str
    applicable: bool
    decision: str | None
    trace: list[TraceStepResponse]
    obligations: list[ObligationResponse]
    source: str | None = None
    notes: str | None = None


class DecideResponse(BaseModel):
    """Response for regulatory decision (may include multiple rules)."""

    results: list[DecisionResponse]
    summary: str | None = Field(None, description="Summary of overall outcome")


# =============================================================================
# Rules Models
# =============================================================================


class RuleInfo(BaseModel):
    """Summary information about a rule."""

    rule_id: str
    version: str
    description: str | None
    effective_from: str | None
    effective_to: str | None
    tags: list[str]
    source: str | None


class RulesListResponse(BaseModel):
    """Response listing available rules."""

    rules: list[RuleInfo]
    total: int


class RuleDetailResponse(BaseModel):
    """Detailed rule information."""

    rule_id: str
    version: str
    description: str | None
    effective_from: str | None
    effective_to: str | None
    tags: list[str]
    source: dict | None
    applies_if: dict | None
    decision_tree: dict | None
    interpretation_notes: str | None
