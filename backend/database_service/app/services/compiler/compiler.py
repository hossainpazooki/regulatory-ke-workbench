"""
Rule compiler for transforming AST to IR.

Compiles rules from their YAML/Pydantic representation to an optimized
Intermediate Representation for efficient runtime execution.

Extended with jurisdiction support for v4 multi-jurisdiction architecture.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from backend.rule_service.app.services.loader import (
    Rule,
    ConditionGroupSpec,
    ConditionSpec,
    DecisionNode,
    DecisionLeaf,
    ObligationSpec as RuleObligationSpec,
)
from .ir import (
    CompiledCheck,
    DecisionEntry,
    ObligationSpec,
    RuleIR,
)
from backend.core.ontology.jurisdiction import (
    Jurisdiction,
    JurisdictionCode,
    JURISDICTION_NAMES,
    JURISDICTION_AUTHORITIES,
)


# Operator mapping from YAML to IR
OPERATOR_MAP = {
    "==": "eq",
    "=": "eq",
    "!=": "ne",
    "<>": "ne",
    "in": "in",
    "not in": "not_in",
    "not_in": "not_in",
    ">": "gt",
    "<": "lt",
    ">=": "gte",
    "<=": "lte",
    "exists": "exists",
}


class RuleCompiler:
    """Compiles rules from AST to IR."""

    def __init__(self):
        self._check_index = 0
        self._decision_index = 0
        self._entry_id = 0

    def compile(self, rule: Rule, yaml_content: str | None = None) -> RuleIR:
        """Compile a rule to IR.

        Implements v4 pretreat layer optimizations:
        - buildPremiseIndex() with jurisdiction keys
        - flattenConditions() for linear evaluation
        - precomputeValueSets() for O(1) 'in' checks
        - generateDecisionTable() for jump table lookup
        - extractAllObligations() for fast access

        Args:
            rule: The Rule object to compile
            yaml_content: Optional YAML source for hash computation

        Returns:
            Compiled RuleIR with jurisdiction support
        """
        # Reset indices for each compilation
        self._check_index = 0
        self._decision_index = 0
        self._entry_id = 0

        # Build jurisdiction object
        jurisdiction_code = getattr(rule, 'jurisdiction', JurisdictionCode.EU)
        if isinstance(jurisdiction_code, str):
            jurisdiction_code = JurisdictionCode(jurisdiction_code)

        jurisdiction = Jurisdiction(
            code=jurisdiction_code,
            name=JURISDICTION_NAMES.get(jurisdiction_code, jurisdiction_code.value),
            authority=JURISDICTION_AUTHORITIES.get(jurisdiction_code, "Unknown"),
        )

        # Get regime_id
        regime_id = getattr(rule, 'regime_id', 'mica_2023')
        cross_border_relevant = getattr(rule, 'cross_border_relevant', False)

        # Extract premise keys for O(1) lookup (includes jurisdiction keys)
        premise_keys = self._extract_premise_keys_with_jurisdiction(
            rule.applies_if, jurisdiction_code, regime_id
        )

        # Flatten applicability conditions
        applicability_checks, applicability_mode = self._flatten_conditions(
            rule.applies_if
        )

        # Generate decision table from decision tree
        decision_checks, decision_table = self._generate_decision_table(
            rule.decision_tree
        )

        # Extract all obligations from decision tree
        all_obligations = self._extract_all_obligations(rule.decision_tree)

        # Compute source hash
        source_hash = None
        if yaml_content:
            source_hash = hashlib.sha256(yaml_content.encode()).hexdigest()[:16]

        return RuleIR(
            rule_id=rule.rule_id,
            version=int(rule.version.split(".")[0]) if "." in rule.version else int(rule.version),
            ir_version=2,
            # Jurisdiction fields
            jurisdiction=jurisdiction,
            jurisdiction_code=jurisdiction_code,
            regime_id=regime_id,
            cross_border_relevant=cross_border_relevant,
            # Premise keys with jurisdiction
            premise_keys=premise_keys,
            # Flattened checks
            applicability_checks=applicability_checks,
            applicability_mode=applicability_mode,
            decision_checks=decision_checks,
            decision_table=decision_table,
            # Pre-extracted obligations
            all_obligations=all_obligations,
            # Metadata
            compiled_at=datetime.now(timezone.utc).isoformat(),
            source_hash=source_hash,
            source_document_id=rule.source.document_id if rule.source else None,
            source_article=rule.source.article if rule.source else None,
        )

    def _extract_premise_keys(
        self, condition_group: ConditionGroupSpec | None
    ) -> list[str]:
        """Extract premise keys for inverted index.

        Premise keys enable O(1) rule lookup based on fact values.

        Args:
            condition_group: The applies_if condition group

        Returns:
            List of premise keys in format "field:value"
        """
        if not condition_group:
            return []

        keys: list[str] = []

        def process_condition(cond: ConditionSpec) -> None:
            field = cond.field
            value = cond.value
            operator = cond.operator

            if operator in ("==", "="):
                keys.append(f"{field}:{value}")
            elif operator == "in" and isinstance(value, list):
                for v in value:
                    keys.append(f"{field}:{v}")
            elif operator == "exists":
                keys.append(f"{field}:*")

        def process_group(group: ConditionGroupSpec) -> None:
            conditions = group.all or group.any or []
            for item in conditions:
                if isinstance(item, ConditionSpec):
                    process_condition(item)
                elif isinstance(item, ConditionGroupSpec):
                    process_group(item)

        process_group(condition_group)
        return list(set(keys))

    def _extract_premise_keys_with_jurisdiction(
        self,
        condition_group: ConditionGroupSpec | None,
        jurisdiction_code: JurisdictionCode,
        regime_id: str,
    ) -> list[str]:
        """Extract premise keys with jurisdiction keys for O(1) filtered lookup.

        CRITICAL for v4 architecture: Always includes jurisdiction and regime keys
        to enable O(1) filtered lookup across jurisdictions.

        Args:
            condition_group: The applies_if condition group
            jurisdiction_code: The rule's jurisdiction
            regime_id: The regulatory regime identifier

        Returns:
            List of premise keys including jurisdiction keys
        """
        keys: list[str] = []

        # Always add jurisdiction and regime keys (v4 requirement)
        keys.append(f"jurisdiction:{jurisdiction_code.value}")
        keys.append(f"regime:{regime_id}")

        # Extract from condition group
        if condition_group:
            keys.extend(self._extract_premise_keys(condition_group))

        return list(set(keys))

    def _extract_all_obligations(
        self, tree: DecisionNode | DecisionLeaf | None
    ) -> list[dict]:
        """Extract all unique obligations from decision tree.

        Pre-extracts obligations for fast access at runtime.

        Args:
            tree: The decision tree root

        Returns:
            List of obligation dicts
        """
        if tree is None:
            return []

        obligations: list[dict] = []
        seen_ids: set[str] = set()

        def traverse(node: DecisionNode | DecisionLeaf) -> None:
            if isinstance(node, DecisionLeaf):
                if node.obligations:
                    for obl in node.obligations:
                        if obl.id not in seen_ids:
                            seen_ids.add(obl.id)
                            obligations.append({
                                "id": obl.id,
                                "description": obl.description,
                                "deadline": obl.deadline,
                                "source_ref": getattr(obl, "source_ref", None),
                            })
            else:
                # Decision node
                if node.true_branch:
                    traverse(node.true_branch)
                if node.false_branch:
                    traverse(node.false_branch)

        traverse(tree)
        return obligations

    def _flatten_conditions(
        self, condition_group: ConditionGroupSpec | None
    ) -> tuple[list[CompiledCheck], str]:
        """Flatten nested conditions to linear check sequence.

        Args:
            condition_group: The condition group to flatten

        Returns:
            Tuple of (list of CompiledCheck, mode 'all' or 'any')
        """
        if not condition_group:
            return [], "all"

        checks: list[CompiledCheck] = []
        mode = "all" if condition_group.all else "any"
        conditions = condition_group.all or condition_group.any or []

        for item in conditions:
            if isinstance(item, ConditionSpec):
                check = self._compile_condition(item)
                checks.append(check)
            elif isinstance(item, ConditionGroupSpec):
                # For nested groups, recursively flatten
                # Note: Full support for arbitrary nesting would require
                # more complex control flow; this handles simple cases
                nested_checks, _ = self._flatten_conditions(item)
                checks.extend(nested_checks)

        return checks, mode

    def _compile_condition(self, cond: ConditionSpec) -> CompiledCheck:
        """Compile a single condition to a check.

        Args:
            cond: The condition specification

        Returns:
            Compiled check
        """
        op = OPERATOR_MAP.get(cond.operator, cond.operator)

        # Pre-compute value set for 'in' operator
        value_set = None
        if op == "in" and isinstance(cond.value, list):
            value_set = set(str(v) for v in cond.value)

        check = CompiledCheck(
            index=self._check_index,
            field=cond.field,
            op=op,
            value=cond.value,
            value_set=value_set,
        )
        self._check_index += 1
        return check

    def _generate_decision_table(
        self, tree: DecisionNode | DecisionLeaf | None
    ) -> tuple[list[CompiledCheck], list[DecisionEntry]]:
        """Generate decision table from decision tree.

        Walks the tree and generates:
        1. A list of decision checks (conditions in the tree)
        2. A decision table with entries for each possible path

        Args:
            tree: The decision tree root

        Returns:
            Tuple of (decision checks, decision table entries)
        """
        if tree is None:
            return [], []

        checks: list[CompiledCheck] = []
        entries: list[DecisionEntry] = []

        # Collect all paths through the tree
        self._walk_tree(tree, [], checks, entries)

        return checks, entries

    def _walk_tree(
        self,
        node: DecisionNode | DecisionLeaf,
        path: list[tuple[int, bool]],
        checks: list[CompiledCheck],
        entries: list[DecisionEntry],
    ) -> None:
        """Recursively walk decision tree to generate table entries.

        Args:
            node: Current node in the tree
            path: Current path as list of (check_index, is_true_branch)
            checks: Accumulator for decision checks
            entries: Accumulator for decision entries
        """
        if isinstance(node, DecisionLeaf):
            # Leaf node: create decision entry
            condition_mask = self._path_to_mask(path, len(checks))
            obligations = [
                ObligationSpec(
                    id=o.id,
                    description=o.description,
                    deadline=o.deadline,
                )
                for o in (node.obligations or [])
            ]
            entry = DecisionEntry(
                entry_id=self._entry_id,
                condition_mask=condition_mask,
                result=node.result,
                obligations=obligations,
                notes=node.notes,
            )
            self._entry_id += 1
            entries.append(entry)
            return

        # Decision node: add check and recurse
        if node.condition:
            check_idx = self._decision_index
            check = CompiledCheck(
                index=check_idx,
                field=node.condition.field,
                op=OPERATOR_MAP.get(node.condition.operator, node.condition.operator),
                value=node.condition.value,
            )

            # Pre-compute value set for 'in' operator
            if check.op == "in" and isinstance(node.condition.value, list):
                check.value_set = set(str(v) for v in node.condition.value)

            checks.append(check)
            self._decision_index += 1

            # Recurse into true branch
            if node.true_branch:
                self._walk_tree(
                    node.true_branch, path + [(check_idx, True)], checks, entries
                )

            # Recurse into false branch
            if node.false_branch:
                self._walk_tree(
                    node.false_branch, path + [(check_idx, False)], checks, entries
                )
        else:
            # No condition, just traverse branches
            if node.true_branch:
                self._walk_tree(node.true_branch, path, checks, entries)
            if node.false_branch:
                self._walk_tree(node.false_branch, path, checks, entries)

    def _path_to_mask(
        self, path: list[tuple[int, bool]], total_checks: int
    ) -> list[int]:
        """Convert path to condition mask.

        Args:
            path: List of (check_index, is_true_branch) tuples
            total_checks: Total number of checks

        Returns:
            Condition mask where:
            - +i means check i must be True
            - -i means check i must be False
            - 0 means don't care
        """
        mask = [0] * total_checks

        for check_idx, is_true in path:
            if check_idx < total_checks:
                # Use 1-based indexing in mask to distinguish from 0 (don't care)
                mask[check_idx] = (check_idx + 1) if is_true else -(check_idx + 1)

        return mask


def compile_rule(rule: Rule, yaml_content: str | None = None) -> RuleIR:
    """Convenience function to compile a single rule.

    Args:
        rule: The rule to compile
        yaml_content: Optional YAML source

    Returns:
        Compiled RuleIR
    """
    compiler = RuleCompiler()
    return compiler.compile(rule, yaml_content)


def compile_rules(rules: list[Rule]) -> dict[str, RuleIR]:
    """Compile multiple rules.

    Args:
        rules: List of rules to compile

    Returns:
        Dict mapping rule_id to RuleIR
    """
    compiler = RuleCompiler()
    return {rule.rule_id: compiler.compile(rule) for rule in rules}
