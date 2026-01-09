"""YAML rule loader and validator.

Extended with jurisdiction support for v4 multi-jurisdiction architecture.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .schema import (
    ConsistencyBlock,
    ConsistencySummary,
    ConsistencyEvidence,
    ConsistencyStatus,
)
from backend.core.ontology.jurisdiction import JurisdictionCode


class SourceRef(BaseModel):
    """Source reference in a rule."""

    document_id: str
    article: str | None = None
    section: str | None = None
    paragraphs: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    url: str | None = None


class ConditionSpec(BaseModel):
    """A single condition specification."""

    field: str
    operator: str = "=="
    value: Any = None


class ConditionGroupSpec(BaseModel):
    """Grouped conditions (all/any)."""

    all: list[ConditionSpec | ConditionGroupSpec] | None = None
    any: list[ConditionSpec | ConditionGroupSpec] | None = None


class ObligationSpec(BaseModel):
    """An obligation triggered by a rule."""

    id: str
    description: str | None = None
    deadline: str | None = None


class DecisionNode(BaseModel):
    """A node in the decision tree."""

    node_id: str
    condition: ConditionSpec | None = None
    true_branch: DecisionNode | DecisionLeaf | None = None
    false_branch: DecisionNode | DecisionLeaf | None = None


class DecisionLeaf(BaseModel):
    """A leaf node (terminal result) in the decision tree."""

    result: str
    obligations: list[ObligationSpec] = Field(default_factory=list)
    notes: str | None = None


class Rule(BaseModel):
    """A complete rule specification.

    Extended with jurisdiction support for v4 architecture.
    """

    rule_id: str
    version: str = "1.0"
    description: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None

    # Jurisdiction scoping (v4 multi-jurisdiction support)
    jurisdiction: JurisdictionCode = JurisdictionCode.EU
    regime_id: str = "mica_2023"
    cross_border_relevant: bool = False

    applies_if: ConditionGroupSpec | None = None
    decision_tree: DecisionNode | DecisionLeaf | None = None

    source: SourceRef | None = None
    interpretation_notes: str | None = None
    tags: list[str] = Field(default_factory=list)

    # QA / Consistency metadata
    consistency: ConsistencyBlock | None = None


# Enable forward references
ConditionGroupSpec.model_rebuild()
DecisionNode.model_rebuild()


class RuleLoader:
    """Loads and validates YAML rules from files or directories."""

    def __init__(self, rules_dir: str | Path | None = None):
        self.rules_dir = Path(rules_dir) if rules_dir else None
        self._rules: dict[str, Rule] = {}

    def load_file(self, path: str | Path) -> list[Rule]:
        """Load rules from a single YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Rule file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)

        rules = []
        # Handle single rule or list of rules
        if isinstance(content, list):
            for item in content:
                rule = self._parse_rule(item)
                rules.append(rule)
                self._rules[rule.rule_id] = rule
        else:
            rule = self._parse_rule(content)
            rules.append(rule)
            self._rules[rule.rule_id] = rule

        return rules

    def load_directory(self, path: str | Path | None = None) -> list[Rule]:
        """Load all YAML rules from a directory."""
        path = Path(path) if path else self.rules_dir
        if not path:
            raise ValueError("No rules directory specified")
        if not path.exists():
            raise FileNotFoundError(f"Rules directory not found: {path}")

        rules = []
        for yaml_file in path.glob("*.yaml"):
            # Skip schema file
            if yaml_file.name == "schema.yaml":
                continue
            try:
                rules.extend(self.load_file(yaml_file))
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

        return rules

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get a loaded rule by ID."""
        return self._rules.get(rule_id)

    def get_all_rules(self) -> list[Rule]:
        """Get all loaded rules."""
        return list(self._rules.values())

    def get_applicable_rules(self, tags: list[str] | None = None) -> list[Rule]:
        """Get rules filtered by tags and effective date."""
        today = date.today()
        rules = []

        for rule in self._rules.values():
            # Check effective dates
            if rule.effective_from and rule.effective_from > today:
                continue
            if rule.effective_to and rule.effective_to < today:
                continue

            # Check tags
            if tags:
                if not any(tag in rule.tags for tag in tags):
                    continue

            rules.append(rule)

        return rules

    def _parse_rule(self, data: dict) -> Rule:
        """Parse a rule from dictionary data."""
        # Convert applies_if
        if "applies_if" in data and data["applies_if"]:
            data["applies_if"] = self._parse_condition_group(data["applies_if"])

        # Convert decision_tree
        if "decision_tree" in data and data["decision_tree"]:
            data["decision_tree"] = self._parse_decision_node(data["decision_tree"])

        # Convert consistency block if present
        if "consistency" in data and data["consistency"]:
            data["consistency"] = self._parse_consistency(data["consistency"])

        return Rule(**data)

    def _parse_consistency(self, data: dict) -> ConsistencyBlock:
        """Parse a consistency block."""
        summary_data = data.get("summary", {})
        summary = ConsistencySummary(
            status=ConsistencyStatus(summary_data.get("status", "unverified")),
            confidence=summary_data.get("confidence", 0.0),
            last_verified=summary_data.get("last_verified"),
            verified_by=summary_data.get("verified_by"),
            notes=summary_data.get("notes"),
        )

        evidence = []
        for ev_data in data.get("evidence", []):
            evidence.append(ConsistencyEvidence(
                tier=ev_data.get("tier", 0),
                category=ev_data.get("category", "unknown"),
                label=ev_data.get("label", "warning"),
                score=ev_data.get("score", 0.0),
                details=ev_data.get("details", ""),
                source_span=ev_data.get("source_span"),
                rule_element=ev_data.get("rule_element"),
                timestamp=ev_data.get("timestamp", ""),
            ))

        return ConsistencyBlock(summary=summary, evidence=evidence)

    def _parse_condition_group(self, data: dict) -> ConditionGroupSpec:
        """Parse a condition group."""
        result = {}

        if "all" in data:
            result["all"] = [
                self._parse_condition_or_group(c) for c in data["all"]
            ]
        if "any" in data:
            result["any"] = [
                self._parse_condition_or_group(c) for c in data["any"]
            ]

        return ConditionGroupSpec(**result)

    def _parse_condition_or_group(
        self, data: dict
    ) -> ConditionSpec | ConditionGroupSpec:
        """Parse either a condition or a condition group."""
        if "field" in data:
            return ConditionSpec(**data)
        return self._parse_condition_group(data)

    def _parse_decision_node(self, data: dict) -> DecisionNode | DecisionLeaf:
        """Parse a decision tree node or leaf."""
        # Check if it's a leaf node
        if "result" in data:
            obligations = []
            if "obligations" in data:
                obligations = [ObligationSpec(**o) for o in data["obligations"]]
            return DecisionLeaf(
                result=data["result"],
                obligations=obligations,
                notes=data.get("notes"),
            )

        # It's a branch node
        condition = None
        if "condition" in data:
            cond_data = data["condition"]
            if isinstance(cond_data, dict):
                condition = ConditionSpec(**cond_data)
            elif isinstance(cond_data, str):
                # Parse string condition like "authorized == true"
                condition = self._parse_string_condition(cond_data)

        true_branch = None
        false_branch = None
        if "true_branch" in data and data["true_branch"]:
            true_branch = self._parse_decision_node(data["true_branch"])
        if "false_branch" in data and data["false_branch"]:
            false_branch = self._parse_decision_node(data["false_branch"])

        return DecisionNode(
            node_id=data.get("node_id", "unnamed"),
            condition=condition,
            true_branch=true_branch,
            false_branch=false_branch,
        )

    def _parse_string_condition(self, cond_str: str) -> ConditionSpec:
        """Parse a string condition like 'authorized == true'."""
        operators = ["==", "!=", ">=", "<=", ">", "<", " in "]
        for op in operators:
            if op in cond_str:
                parts = cond_str.split(op)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value_str = parts[1].strip()
                    value = self._parse_value(value_str)
                    return ConditionSpec(field=field, operator=op.strip(), value=value)

        # Default: treat as existence check
        return ConditionSpec(field=cond_str.strip(), operator="exists", value=True)

    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value into appropriate type."""
        # Boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # List
        if value_str.startswith("[") and value_str.endswith("]"):
            inner = value_str[1:-1]
            items = [item.strip().strip("'\"") for item in inner.split(",")]
            return items

        # Number
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # String (remove quotes if present)
        return value_str.strip("'\"")

    def save_rule(self, rule: Rule, path: str | Path | None = None) -> Path:
        """Save a rule to a YAML file.

        Args:
            rule: The rule to save.
            path: Optional path. If not provided, saves to rules_dir/{rule_id}.yaml

        Returns:
            Path to the saved file.
        """
        if path is None:
            if self.rules_dir is None:
                raise ValueError("No rules directory specified and no path provided")
            path = self.rules_dir / f"{rule.rule_id}.yaml"
        else:
            path = Path(path)

        # Convert rule to dict, handling special types
        data = self._rule_to_dict(rule)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Update internal cache
        self._rules[rule.rule_id] = rule

        return path

    def _rule_to_dict(self, rule: Rule) -> dict:
        """Convert a Rule to a dictionary suitable for YAML serialization."""
        # Use mode='json' to get JSON-serializable output (handles enums, dates, etc.)
        data = rule.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        return data

    def update_rule(self, rule_id: str, updates: dict) -> Rule:
        """Update a rule with new values and save.

        Args:
            rule_id: ID of the rule to update.
            updates: Dictionary of fields to update.

        Returns:
            The updated rule.
        """
        rule = self.get_rule(rule_id)
        if not rule:
            raise ValueError(f"Rule not found: {rule_id}")

        # Create updated rule
        rule_data = rule.model_dump()
        rule_data.update(updates)

        # Re-parse to validate
        updated_rule = Rule(**rule_data)
        self._rules[rule_id] = updated_rule

        return updated_rule

    def validate_corpus_coverage(self) -> dict[str, list[str]]:
        """Validate that rule document_ids have corresponding legal corpus entries.

        Returns:
            Dict with 'valid', 'missing', and 'warnings' lists of rule_ids.
        """
        from backend.rag_service.app.services.corpus_loader import get_available_document_ids

        available_docs = set(get_available_document_ids())
        results = {
            "valid": [],
            "missing": [],
            "warnings": [],
        }

        for rule in self._rules.values():
            if not rule.source:
                results["warnings"].append(rule.rule_id)
                continue

            doc_id = rule.source.document_id
            if doc_id in available_docs:
                results["valid"].append(rule.rule_id)
            else:
                results["missing"].append(rule.rule_id)

        return results

    def get_rules_by_document(self, document_id: str) -> list[Rule]:
        """Get all rules that reference a specific document.

        Args:
            document_id: The document identifier to filter by.

        Returns:
            List of rules with matching source.document_id.
        """
        return [
            rule for rule in self._rules.values()
            if rule.source and rule.source.document_id == document_id
        ]
