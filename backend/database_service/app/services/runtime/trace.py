"""
Execution tracing for runtime rule evaluation.

Provides detailed traces of how decisions are made, enabling:
- Audit trails for regulatory compliance
- Debugging and explanation generation
- Backward compatibility with existing TraceStep format
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    """A single step in the execution trace.

    Compatible with the existing TraceStep format used by DecisionEngine.
    """

    node_id: str
    """Identifier for this step (e.g., 'check_0', 'applicability')."""

    description: str
    """Human-readable description of what was evaluated."""

    field: str | None = None
    """The fact field that was checked (if applicable)."""

    operator: str | None = None
    """The operator used in the check."""

    expected_value: Any = None
    """The expected value for the check."""

    actual_value: Any = None
    """The actual value from facts."""

    result: bool | None = None
    """Whether the check passed (True/False/None if not evaluated)."""

    source_ref: str | None = None
    """Reference to source legal text."""


class ExecutionTrace(BaseModel):
    """Complete execution trace for a rule evaluation.

    Contains all steps taken during rule evaluation, providing
    full transparency for audit and debugging purposes.
    """

    rule_id: str
    """The rule that was evaluated."""

    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    """ISO timestamp of when evaluation started."""

    completed_at: str | None = None
    """ISO timestamp of when evaluation completed."""

    # Applicability evaluation
    applicable: bool = False
    """Whether the rule was applicable to the facts."""

    applicability_steps: list[TraceStep] = Field(default_factory=list)
    """Steps taken during applicability evaluation."""

    # Decision evaluation
    decision: str | None = None
    """The final decision result."""

    decision_steps: list[TraceStep] = Field(default_factory=list)
    """Steps taken during decision tree evaluation."""

    # Metadata
    facts_used: dict[str, Any] = Field(default_factory=dict)
    """The fact values that were actually used during evaluation."""

    entry_matched: int | None = None
    """Index of the decision table entry that matched (if any)."""

    obligations: list[dict[str, Any]] = Field(default_factory=list)
    """Obligations triggered by the decision."""

    def add_applicability_step(
        self,
        node_id: str,
        description: str,
        field: str | None = None,
        operator: str | None = None,
        expected_value: Any = None,
        actual_value: Any = None,
        result: bool | None = None,
    ) -> TraceStep:
        """Add a step to applicability trace.

        Returns:
            The created TraceStep
        """
        step = TraceStep(
            node_id=node_id,
            description=description,
            field=field,
            operator=operator,
            expected_value=expected_value,
            actual_value=actual_value,
            result=result,
        )
        self.applicability_steps.append(step)
        return step

    def add_decision_step(
        self,
        node_id: str,
        description: str,
        field: str | None = None,
        operator: str | None = None,
        expected_value: Any = None,
        actual_value: Any = None,
        result: bool | None = None,
        source_ref: str | None = None,
    ) -> TraceStep:
        """Add a step to decision trace.

        Returns:
            The created TraceStep
        """
        step = TraceStep(
            node_id=node_id,
            description=description,
            field=field,
            operator=operator,
            expected_value=expected_value,
            actual_value=actual_value,
            result=result,
            source_ref=source_ref,
        )
        self.decision_steps.append(step)
        return step

    def complete(self, decision: str | None = None) -> None:
        """Mark the trace as complete.

        Args:
            decision: The final decision result
        """
        self.decision = decision
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_legacy_trace(self) -> list[dict[str, Any]]:
        """Convert to legacy trace format for backward compatibility.

        Returns:
            List of trace step dictionaries in legacy format
        """
        legacy = []

        for step in self.applicability_steps:
            legacy.append({
                "node_id": step.node_id,
                "description": step.description,
                "field": step.field,
                "operator": step.operator,
                "expected_value": step.expected_value,
                "actual_value": step.actual_value,
                "result": step.result,
            })

        for step in self.decision_steps:
            legacy.append({
                "node_id": step.node_id,
                "description": step.description,
                "field": step.field,
                "operator": step.operator,
                "expected_value": step.expected_value,
                "actual_value": step.actual_value,
                "result": step.result,
            })

        return legacy


class DecisionResult(BaseModel):
    """Result of a rule evaluation.

    Provides a clean interface for decision outcomes.
    """

    rule_id: str
    """The rule that was evaluated."""

    applicable: bool
    """Whether the rule was applicable to the facts."""

    decision: str | None = None
    """The final decision (None if not applicable)."""

    obligations: list[dict[str, Any]] = Field(default_factory=list)
    """Obligations triggered by the decision."""

    trace: ExecutionTrace | None = None
    """Detailed execution trace (optional)."""

    @classmethod
    def not_applicable(cls, rule_id: str, trace: ExecutionTrace | None = None) -> "DecisionResult":
        """Create a result for when a rule is not applicable.

        Args:
            rule_id: The rule ID
            trace: Optional execution trace

        Returns:
            DecisionResult with applicable=False
        """
        return cls(
            rule_id=rule_id,
            applicable=False,
            trace=trace,
        )

    @classmethod
    def with_decision(
        cls,
        rule_id: str,
        decision: str,
        obligations: list[dict[str, Any]] | None = None,
        trace: ExecutionTrace | None = None,
    ) -> "DecisionResult":
        """Create a result with a decision.

        Args:
            rule_id: The rule ID
            decision: The decision result
            obligations: List of obligations
            trace: Optional execution trace

        Returns:
            DecisionResult with the decision
        """
        return cls(
            rule_id=rule_id,
            applicable=True,
            decision=decision,
            obligations=obligations or [],
            trace=trace,
        )
