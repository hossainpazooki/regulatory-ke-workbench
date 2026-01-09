"""Decision engine with trace generation."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from backend.core.ontology.scenario import Scenario
from .loader import (
    Rule,
    RuleLoader,
    ConditionSpec,
    ConditionGroupSpec,
    DecisionNode,
    DecisionLeaf,
    SourceRef,
)
from .schema import ConsistencyBlock, ConsistencySummary


class ObligationResult(BaseModel):
    """An obligation from a decision."""

    id: str
    description: str | None = None
    source: str | None = None
    deadline: str | None = None


class TraceStep(BaseModel):
    """A single step in the decision trace."""

    node: str
    condition: str
    result: bool
    value_checked: Any = None


class RuleMetadata(BaseModel):
    """Metadata about the rule that produced a decision."""

    rule_id: str
    version: str = "1.0"
    source: SourceRef | None = None
    consistency: ConsistencyBlock | None = None
    tags: list[str] = Field(default_factory=list)


class DecisionResult(BaseModel):
    """Complete result of a decision evaluation."""

    rule_id: str
    applicable: bool = True
    decision: str | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    obligations: list[ObligationResult] = Field(default_factory=list)
    source: str | None = None
    notes: str | None = None

    # Full rule metadata including consistency
    rule_metadata: RuleMetadata | None = None


class DecisionEngine:
    """Evaluates rules against scenarios with full tracing."""

    def __init__(self, loader: RuleLoader | None = None):
        self.loader = loader or RuleLoader()

    def evaluate(self, scenario: Scenario, rule_id: str) -> DecisionResult:
        """Evaluate a specific rule against a scenario."""
        rule = self.loader.get_rule(rule_id)
        if not rule:
            return DecisionResult(
                rule_id=rule_id,
                applicable=False,
                decision="rule_not_found",
            )

        return self._evaluate_rule(rule, scenario)

    def evaluate_all(self, scenario: Scenario) -> list[DecisionResult]:
        """Evaluate all applicable rules against a scenario."""
        results = []
        for rule in self.loader.get_all_rules():
            result = self._evaluate_rule(rule, scenario)
            if result.applicable:
                results.append(result)
        return results

    def _evaluate_rule(self, rule: Rule, scenario: Scenario) -> DecisionResult:
        """Evaluate a single rule against a scenario."""
        context = scenario.to_flat_dict()
        trace: list[TraceStep] = []

        # Build source string
        source_str = None
        if rule.source:
            parts = [rule.source.document_id]
            if rule.source.article:
                parts.append(f"Art. {rule.source.article}")
            if rule.source.pages:
                parts.append(f"p. {', '.join(map(str, rule.source.pages))}")
            source_str = " ".join(parts)

        # Build rule metadata
        rule_metadata = RuleMetadata(
            rule_id=rule.rule_id,
            version=rule.version,
            source=rule.source,
            consistency=rule.consistency,
            tags=rule.tags,
        )

        # Check applies_if conditions
        if rule.applies_if:
            applies, applies_trace = self._evaluate_condition_group(
                rule.applies_if, context, "applicability"
            )
            trace.extend(applies_trace)

            if not applies:
                return DecisionResult(
                    rule_id=rule.rule_id,
                    applicable=False,
                    decision="not_applicable",
                    trace=trace,
                    source=source_str,
                    rule_metadata=rule_metadata,
                )

        # Evaluate decision tree
        if rule.decision_tree:
            decision, obligations, tree_trace = self._evaluate_decision_tree(
                rule.decision_tree, context, rule.source
            )
            trace.extend(tree_trace)

            return DecisionResult(
                rule_id=rule.rule_id,
                applicable=True,
                decision=decision,
                trace=trace,
                obligations=obligations,
                source=source_str,
                notes=rule.interpretation_notes,
                rule_metadata=rule_metadata,
            )

        # Rule has no decision tree (informational only)
        return DecisionResult(
            rule_id=rule.rule_id,
            applicable=True,
            decision="applicable",
            trace=trace,
            source=source_str,
            notes=rule.interpretation_notes,
            rule_metadata=rule_metadata,
        )

    def _evaluate_condition_group(
        self, group: ConditionGroupSpec, context: dict, prefix: str
    ) -> tuple[bool, list[TraceStep]]:
        """Evaluate a condition group (all/any)."""
        trace = []

        if group.all:
            for i, cond in enumerate(group.all):
                if isinstance(cond, ConditionGroupSpec):
                    result, sub_trace = self._evaluate_condition_group(
                        cond, context, f"{prefix}.all[{i}]"
                    )
                    trace.extend(sub_trace)
                else:
                    result, step = self._evaluate_condition(cond, context, f"{prefix}.all[{i}]")
                    trace.append(step)

                if not result:
                    return False, trace

            return True, trace

        if group.any:
            for i, cond in enumerate(group.any):
                if isinstance(cond, ConditionGroupSpec):
                    result, sub_trace = self._evaluate_condition_group(
                        cond, context, f"{prefix}.any[{i}]"
                    )
                    trace.extend(sub_trace)
                else:
                    result, step = self._evaluate_condition(cond, context, f"{prefix}.any[{i}]")
                    trace.append(step)

                if result:
                    return True, trace

            return False, trace

        return True, trace

    def _evaluate_condition(
        self, cond: ConditionSpec, context: dict, node_id: str
    ) -> tuple[bool, TraceStep]:
        """Evaluate a single condition."""
        field = cond.field
        op = cond.operator
        expected = cond.value
        actual = context.get(field)

        condition_str = f"{field} {op} {expected}"
        result = False

        if op == "==":
            result = actual == expected
        elif op == "!=":
            result = actual != expected
        elif op == "in":
            if isinstance(expected, list):
                result = actual in expected
            else:
                result = False
        elif op == "not_in":
            if isinstance(expected, list):
                result = actual not in expected
            else:
                result = True
        elif op == ">":
            result = actual is not None and actual > expected
        elif op == "<":
            result = actual is not None and actual < expected
        elif op == ">=":
            result = actual is not None and actual >= expected
        elif op == "<=":
            result = actual is not None and actual <= expected
        elif op == "exists":
            result = actual is not None

        step = TraceStep(
            node=node_id,
            condition=condition_str,
            result=result,
            value_checked=actual,
        )
        return result, step

    def _evaluate_decision_tree(
        self, node: DecisionNode | DecisionLeaf, context: dict, source
    ) -> tuple[str, list[ObligationResult], list[TraceStep]]:
        """Evaluate a decision tree."""
        trace = []

        # Leaf node
        if isinstance(node, DecisionLeaf):
            obligations = []
            for obl in node.obligations:
                obl_source = None
                if source:
                    parts = [source.document_id]
                    if source.article:
                        parts.append(f"Art. {source.article}")
                    if source.pages:
                        parts.append(f"p. {', '.join(map(str, source.pages))}")
                    obl_source = " ".join(parts)

                obligations.append(
                    ObligationResult(
                        id=obl.id,
                        description=obl.description,
                        source=obl_source,
                        deadline=obl.deadline,
                    )
                )
            return node.result, obligations, trace

        # Branch node
        if node.condition:
            result, step = self._evaluate_condition(
                node.condition, context, node.node_id
            )
            trace.append(step)

            next_node = node.true_branch if result else node.false_branch
            if next_node:
                decision, obligations, sub_trace = self._evaluate_decision_tree(
                    next_node, context, source
                )
                trace.extend(sub_trace)
                return decision, obligations, trace

        return "no_decision", [], trace
