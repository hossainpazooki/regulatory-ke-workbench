"""
Runtime executor for compiled rule IR.

Provides efficient O(1) rule lookup and linear condition evaluation.
"""

from __future__ import annotations

from typing import Any

from backend.database_service.app.services.compiler.ir import RuleIR, CompiledCheck, DecisionEntry
from backend.database_service.app.services.compiler.premise_index import PremiseIndexBuilder
from .cache import IRCache, get_ir_cache
from .trace import ExecutionTrace, DecisionResult


# Operator implementations
def _eval_eq(actual: Any, expected: Any) -> bool:
    """Evaluate equality check."""
    return actual == expected


def _eval_ne(actual: Any, expected: Any) -> bool:
    """Evaluate not-equal check."""
    return actual != expected


def _eval_in(actual: Any, expected: Any, value_set: set[str] | None = None) -> bool:
    """Evaluate 'in' check with optional pre-computed set."""
    if value_set is not None:
        return str(actual) in value_set
    if isinstance(expected, (list, tuple, set)):
        return actual in expected
    return False


def _eval_not_in(actual: Any, expected: Any, value_set: set[str] | None = None) -> bool:
    """Evaluate 'not in' check."""
    return not _eval_in(actual, expected, value_set)


def _eval_gt(actual: Any, expected: Any) -> bool:
    """Evaluate greater-than check."""
    try:
        return actual > expected
    except TypeError:
        return False


def _eval_lt(actual: Any, expected: Any) -> bool:
    """Evaluate less-than check."""
    try:
        return actual < expected
    except TypeError:
        return False


def _eval_gte(actual: Any, expected: Any) -> bool:
    """Evaluate greater-than-or-equal check."""
    try:
        return actual >= expected
    except TypeError:
        return False


def _eval_lte(actual: Any, expected: Any) -> bool:
    """Evaluate less-than-or-equal check."""
    try:
        return actual <= expected
    except TypeError:
        return False


def _eval_exists(actual: Any, expected: Any) -> bool:
    """Evaluate existence check."""
    return actual is not None


OPERATORS = {
    "eq": _eval_eq,
    "ne": _eval_ne,
    "in": _eval_in,
    "not_in": _eval_not_in,
    "gt": _eval_gt,
    "lt": _eval_lt,
    "gte": _eval_gte,
    "lte": _eval_lte,
    "exists": _eval_exists,
}


class RuleRuntime:
    """Runtime executor for compiled rules.

    Provides efficient rule evaluation using:
    - O(1) rule lookup via premise index
    - Linear condition evaluation (no tree traversal)
    - In-memory IR caching
    """

    def __init__(
        self,
        cache: IRCache | None = None,
        premise_index: PremiseIndexBuilder | None = None,
    ):
        """Initialize the runtime.

        Args:
            cache: IR cache instance (uses global if not provided)
            premise_index: Premise index for rule lookup
        """
        self._cache = cache or get_ir_cache()
        self._premise_index = premise_index or PremiseIndexBuilder()

    def load_ir(self, rule_id: str, ir_json: str | None = None) -> RuleIR | None:
        """Load a rule IR, using cache if available.

        Args:
            rule_id: The rule identifier
            ir_json: Optional IR JSON to load (if not in cache)

        Returns:
            RuleIR if available
        """
        # Check cache first
        ir = self._cache.get(rule_id)
        if ir is not None:
            return ir

        # Parse from JSON if provided
        if ir_json:
            ir = RuleIR.from_json(ir_json)
            self._cache.put(rule_id, ir)
            return ir

        return None

    def find_applicable_rules(self, facts: dict[str, Any]) -> set[str]:
        """Find all rules that might apply to given facts.

        Uses premise index for O(1) lookup.

        Args:
            facts: Dictionary of fact field -> value

        Returns:
            Set of rule_ids that might apply
        """
        return self._premise_index.lookup(facts)

    def check_applicability(
        self,
        ir: RuleIR,
        facts: dict[str, Any],
        trace: ExecutionTrace | None = None,
    ) -> bool:
        """Check if a rule is applicable to given facts.

        Evaluates applicability_checks linearly.

        Args:
            ir: The compiled RuleIR
            facts: Dictionary of fact values
            trace: Optional trace to record steps

        Returns:
            True if rule is applicable
        """
        if not ir.applicability_checks:
            # No checks = always applicable
            return True

        mode = ir.applicability_mode
        results: list[bool] = []

        for check in ir.applicability_checks:
            result = self._evaluate_check(check, facts)
            results.append(result)

            if trace:
                trace.add_applicability_step(
                    node_id=f"applicability_{check.index}",
                    description=f"Check {check.field} {check.op} {check.value}",
                    field=check.field,
                    operator=check.op,
                    expected_value=check.value,
                    actual_value=facts.get(check.field),
                    result=result,
                )

            # Short-circuit evaluation
            if mode == "all" and not result:
                return False
            if mode == "any" and result:
                return True

        if mode == "all":
            return all(results)
        else:  # mode == "any"
            return any(results)

    def evaluate_decision_table(
        self,
        ir: RuleIR,
        facts: dict[str, Any],
        trace: ExecutionTrace | None = None,
    ) -> DecisionEntry | None:
        """Evaluate decision checks and find matching table entry.

        Args:
            ir: The compiled RuleIR
            facts: Dictionary of fact values
            trace: Optional trace to record steps

        Returns:
            Matching DecisionEntry or None
        """
        if not ir.decision_table:
            return None

        # Evaluate all decision checks
        check_results: list[bool] = []
        for check in ir.decision_checks:
            result = self._evaluate_check(check, facts)
            check_results.append(result)

            if trace:
                trace.add_decision_step(
                    node_id=f"decision_{check.index}",
                    description=f"Check {check.field} {check.op} {check.value}",
                    field=check.field,
                    operator=check.op,
                    expected_value=check.value,
                    actual_value=facts.get(check.field),
                    result=result,
                )

        # Find matching entry in decision table
        for entry in ir.decision_table:
            if self._matches_mask(check_results, entry.condition_mask):
                if trace:
                    trace.add_decision_step(
                        node_id=f"entry_{entry.entry_id}",
                        description=f"Matched entry: {entry.result}",
                        result=True,
                        source_ref=entry.source_ref,
                    )
                    trace.entry_matched = entry.entry_id
                return entry

        return None

    def infer(
        self,
        ir: RuleIR,
        facts: dict[str, Any],
        include_trace: bool = True,
    ) -> DecisionResult:
        """Execute a compiled rule IR against facts.

        Args:
            ir: The compiled RuleIR
            facts: Dictionary of fact values
            include_trace: Whether to include execution trace

        Returns:
            DecisionResult with decision and optional trace
        """
        trace = ExecutionTrace(rule_id=ir.rule_id) if include_trace else None

        if trace:
            trace.facts_used = {
                k: v for k, v in facts.items()
                if any(c.field == k for c in ir.applicability_checks + ir.decision_checks)
            }

        # Check applicability
        applicable = self.check_applicability(ir, facts, trace)

        if trace:
            trace.applicable = applicable

        if not applicable:
            if trace:
                trace.complete()
            return DecisionResult.not_applicable(ir.rule_id, trace)

        # Evaluate decision table
        entry = self.evaluate_decision_table(ir, facts, trace)

        if entry is None:
            if trace:
                trace.complete()
            return DecisionResult.not_applicable(ir.rule_id, trace)

        # Build result
        obligations = [
            {"id": o.id, "description": o.description, "deadline": o.deadline}
            for o in entry.obligations
        ]

        if trace:
            trace.obligations = obligations
            trace.complete(entry.result)

        return DecisionResult.with_decision(
            rule_id=ir.rule_id,
            decision=entry.result,
            obligations=obligations,
            trace=trace,
        )

    def evaluate(
        self,
        rule_id: str,
        facts: dict[str, Any],
        ir_json: str | None = None,
        include_trace: bool = True,
    ) -> DecisionResult | None:
        """Evaluate a rule by ID.

        Convenience method that handles IR loading.

        Args:
            rule_id: The rule identifier
            facts: Dictionary of fact values
            ir_json: Optional IR JSON if not cached
            include_trace: Whether to include execution trace

        Returns:
            DecisionResult or None if rule not found
        """
        ir = self.load_ir(rule_id, ir_json)
        if ir is None:
            return None
        return self.infer(ir, facts, include_trace)

    def evaluate_all(
        self,
        facts: dict[str, Any],
        rule_irs: list[RuleIR] | None = None,
        include_trace: bool = True,
    ) -> list[DecisionResult]:
        """Evaluate all applicable rules against facts.

        Uses premise index for O(1) candidate lookup.

        Args:
            facts: Dictionary of fact values
            rule_irs: Optional list of RuleIRs (uses cached rules if not provided)
            include_trace: Whether to include execution traces

        Returns:
            List of DecisionResults for applicable rules
        """
        results = []

        if rule_irs:
            # Evaluate provided IRs
            for ir in rule_irs:
                result = self.infer(ir, facts, include_trace)
                if result.applicable:
                    results.append(result)
        else:
            # Use premise index to find candidates
            candidates = self.find_applicable_rules(facts)
            for rule_id in candidates:
                ir = self._cache.get(rule_id)
                if ir:
                    result = self.infer(ir, facts, include_trace)
                    if result.applicable:
                        results.append(result)

        return results

    def _evaluate_check(self, check: CompiledCheck, facts: dict[str, Any]) -> bool:
        """Evaluate a single check against facts.

        Args:
            check: The check to evaluate
            facts: Dictionary of fact values

        Returns:
            True if check passes
        """
        actual = facts.get(check.field)
        op_func = OPERATORS.get(check.op, _eval_eq)

        if check.op in ("in", "not_in"):
            return op_func(actual, check.value, check.value_set)
        return op_func(actual, check.value)

    def _matches_mask(
        self,
        check_results: list[bool],
        mask: list[int],
    ) -> bool:
        """Check if evaluation results match a condition mask.

        Args:
            check_results: List of check evaluation results
            mask: Condition mask where:
                  +i means check i-1 must be True
                  -i means check i-1 must be False
                  0 means don't care

        Returns:
            True if results match the mask
        """
        for i, requirement in enumerate(mask):
            if requirement == 0:
                continue  # Don't care

            # Convert 1-based mask index to 0-based check index
            check_idx = abs(requirement) - 1

            if check_idx >= len(check_results):
                continue  # No check at this index

            required_value = requirement > 0
            if check_results[check_idx] != required_value:
                return False

        return True


def execute_rule(
    ir: RuleIR,
    facts: dict[str, Any],
    include_trace: bool = True,
) -> DecisionResult:
    """Convenience function to execute a single rule.

    Args:
        ir: The compiled RuleIR
        facts: Dictionary of fact values
        include_trace: Whether to include execution trace

    Returns:
        DecisionResult
    """
    runtime = RuleRuntime()
    return runtime.infer(ir, facts, include_trace)
