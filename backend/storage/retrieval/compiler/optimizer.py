"""
Optimizer for compiled rules.

Provides optional optimizations for the compiled IR:
- Pre-computation of value sets for O(1) 'in' operator lookups
- Condition reordering by selectivity
- Dead path elimination
"""

from __future__ import annotations

from typing import Any

from .ir import RuleIR, CompiledCheck, DecisionEntry


class RuleOptimizer:
    """Optimizes compiled RuleIR for better runtime performance."""

    def __init__(self, selectivity_hints: dict[str, float] | None = None):
        """Initialize optimizer.

        Args:
            selectivity_hints: Optional dict mapping field names to estimated
                              selectivity (0.0 = very selective, 1.0 = not selective)
        """
        self._selectivity = selectivity_hints or {}

    def optimize(self, ir: RuleIR) -> RuleIR:
        """Apply all optimizations to a RuleIR.

        Args:
            ir: The RuleIR to optimize

        Returns:
            Optimized RuleIR (may be same instance if no changes)
        """
        # Pre-compute value sets
        ir = self._ensure_value_sets(ir)

        # Reorder by selectivity (if hints provided)
        if self._selectivity:
            ir = self._reorder_by_selectivity(ir)

        return ir

    def _ensure_value_sets(self, ir: RuleIR) -> RuleIR:
        """Ensure all 'in' operators have pre-computed value sets.

        Args:
            ir: The RuleIR

        Returns:
            RuleIR with value_set populated for 'in' operators
        """
        for check in ir.applicability_checks:
            if check.op == "in" and check.value_set is None:
                if isinstance(check.value, list):
                    check.value_set = set(str(v) for v in check.value)

        for check in ir.decision_checks:
            if check.op == "in" and check.value_set is None:
                if isinstance(check.value, list):
                    check.value_set = set(str(v) for v in check.value)

        return ir

    def _reorder_by_selectivity(self, ir: RuleIR) -> RuleIR:
        """Reorder applicability checks by selectivity.

        More selective checks (lower selectivity value) are moved first
        for faster short-circuit evaluation.

        Args:
            ir: The RuleIR

        Returns:
            RuleIR with reordered checks
        """
        if ir.applicability_mode != "all":
            # Reordering only helps for AND (all) mode
            return ir

        def get_selectivity(check: CompiledCheck) -> float:
            """Get selectivity estimate for a check."""
            field = check.field
            if field in self._selectivity:
                return self._selectivity[field]

            # Default selectivity heuristics
            if check.op == "eq":
                return 0.1  # Equality is usually selective
            elif check.op == "in":
                # Less selective with more values
                if isinstance(check.value, list):
                    return min(0.5, 0.1 * len(check.value))
                return 0.3
            elif check.op == "exists":
                return 0.8  # Existence is usually not selective
            else:
                return 0.5  # Default

        # Sort by selectivity (most selective first)
        sorted_checks = sorted(
            ir.applicability_checks,
            key=lambda c: get_selectivity(c),
        )

        # Update indices
        for i, check in enumerate(sorted_checks):
            check.index = i

        ir.applicability_checks = sorted_checks
        return ir

    def estimate_selectivity(self, field: str, values: list[Any] | None = None) -> float:
        """Estimate selectivity for a field.

        Args:
            field: Field name
            values: Optional list of values (for 'in' operator)

        Returns:
            Selectivity estimate (0.0 = very selective, 1.0 = not selective)
        """
        if field in self._selectivity:
            return self._selectivity[field]

        # Heuristics based on common field patterns
        if "type" in field.lower():
            return 0.1  # Type fields are usually selective
        elif "id" in field.lower():
            return 0.05  # ID fields are very selective
        elif field.lower() in ("jurisdiction", "country", "region"):
            return 0.2  # Geographic fields are moderately selective
        elif field.lower() in ("active", "enabled", "authorized"):
            return 0.5  # Boolean fields are not very selective
        else:
            return 0.3  # Default

    def analyze_ir(self, ir: RuleIR) -> dict[str, Any]:
        """Analyze a RuleIR for optimization opportunities.

        Args:
            ir: The RuleIR to analyze

        Returns:
            Dict with analysis results
        """
        return {
            "rule_id": ir.rule_id,
            "applicability_check_count": len(ir.applicability_checks),
            "decision_check_count": len(ir.decision_checks),
            "decision_table_size": len(ir.decision_table),
            "premise_key_count": len(ir.premise_keys),
            "has_value_sets": all(
                c.value_set is not None
                for c in ir.applicability_checks + ir.decision_checks
                if c.op == "in"
            ),
            "fields_used": list(
                set(
                    c.field
                    for c in ir.applicability_checks + ir.decision_checks
                )
            ),
        }


def optimize_rule(ir: RuleIR, selectivity_hints: dict[str, float] | None = None) -> RuleIR:
    """Convenience function to optimize a single RuleIR.

    Args:
        ir: The RuleIR to optimize
        selectivity_hints: Optional selectivity hints

    Returns:
        Optimized RuleIR
    """
    optimizer = RuleOptimizer(selectivity_hints)
    return optimizer.optimize(ir)
