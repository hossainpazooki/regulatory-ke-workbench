"""
Migration utilities for loading YAML rules into the database.

Provides functions to migrate existing YAML rule files to the persistence layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

from backend.storage.database import init_db, get_db
from backend.storage.repositories import RuleRepository, VerificationRepository
from backend.rules.service import RuleLoader, Rule, ConditionGroupSpec, ConditionSpec


def extract_premise_keys(condition_group: ConditionGroupSpec | None) -> list[str]:
    """Extract premise keys from a condition group for O(1) lookup index.

    Premise keys are in format "field:value" for equality checks
    or "field:*" for existence checks.

    Args:
        condition_group: The applies_if condition group from a rule

    Returns:
        List of premise key strings
    """
    if not condition_group:
        return []

    keys: list[str] = []

    def process_condition(cond: ConditionSpec) -> None:
        """Extract key from a single condition."""
        field = cond.field
        value = cond.value
        operator = cond.operator

        if operator in ("==", "="):
            # Exact match: field:value
            keys.append(f"{field}:{value}")
        elif operator == "in" and isinstance(value, list):
            # Multiple values: create key for each
            for v in value:
                keys.append(f"{field}:{v}")
        elif operator == "exists":
            keys.append(f"{field}:*")
        # For other operators (!=, <, >, etc.), we don't create index keys

    def process_group(group: ConditionGroupSpec) -> None:
        """Recursively process a condition group."""
        conditions = group.all or group.any or []
        for item in conditions:
            if isinstance(item, ConditionSpec):
                process_condition(item)
            elif isinstance(item, ConditionGroupSpec):
                process_group(item)

    process_group(condition_group)
    return list(set(keys))  # Remove duplicates


def migrate_yaml_rules(
    rules_dir: str | Path,
    clear_existing: bool = False,
) -> dict[str, Any]:
    """Migrate YAML rules from directory to database.

    Args:
        rules_dir: Path to directory containing YAML rule files
        clear_existing: If True, clears existing rules before migration

    Returns:
        Migration result dict with counts and any errors
    """
    # Initialize database
    init_db()

    rule_repo = RuleRepository()
    verify_repo = VerificationRepository()

    # Optionally clear existing data
    if clear_existing:
        with get_db() as conn:
            conn.execute(text("DELETE FROM verification_evidence"))
            conn.execute(text("DELETE FROM verification_results"))
            conn.execute(text("DELETE FROM reviews"))
            conn.execute(text("DELETE FROM rule_premise_index"))
            conn.execute(text("DELETE FROM rules"))
            conn.commit()

    # Load rules from YAML
    loader = RuleLoader(rules_dir)
    try:
        rules = loader.load_directory()
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load rules directory: {e}",
            "rules_migrated": 0,
            "errors": [str(e)],
        }

    result = {
        "success": True,
        "rules_migrated": 0,
        "rules_updated": 0,
        "verifications_migrated": 0,
        "premise_keys_indexed": 0,
        "errors": [],
    }

    for rule in rules:
        try:
            # Read original YAML for storage
            yaml_content = _rule_to_yaml(rule)

            # Check if rule already exists
            existing = rule_repo.get_rule(rule.rule_id)

            # Save rule to database
            rule_repo.save_rule(
                rule_id=rule.rule_id,
                content_yaml=yaml_content,
                source_document_id=rule.source.document_id if rule.source else None,
                source_article=rule.source.article if rule.source else None,
            )

            if existing:
                result["rules_updated"] += 1
            else:
                result["rules_migrated"] += 1

            # Extract and index premise keys
            premise_keys = extract_premise_keys(rule.applies_if)
            if premise_keys:
                rule_repo.update_premise_index(rule.rule_id, premise_keys)
                result["premise_keys_indexed"] += len(premise_keys)

            # Migrate consistency block if present
            if rule.consistency:
                evidence_list = []
                for ev in rule.consistency.evidence:
                    evidence_list.append({
                        "tier": ev.tier,
                        "category": ev.category,
                        "label": ev.label,
                        "score": ev.score,
                        "details": ev.details,
                        "source_span": ev.source_span,
                        "rule_element": ev.rule_element,
                    })

                verify_repo.save_verification_result(
                    rule_id=rule.rule_id,
                    status=rule.consistency.summary.status.value,
                    confidence=rule.consistency.summary.confidence,
                    verified_by=rule.consistency.summary.verified_by,
                    notes=rule.consistency.summary.notes,
                    evidence=evidence_list,
                )
                result["verifications_migrated"] += 1

        except Exception as e:
            result["errors"].append(f"{rule.rule_id}: {e}")

    if result["errors"]:
        result["success"] = len(result["errors"]) < len(rules)

    return result


def _rule_to_yaml(rule: Rule) -> str:
    """Convert a Rule object back to YAML string."""
    data = rule.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def sync_rule_to_db(rule: Rule, rule_repo: RuleRepository | None = None) -> bool:
    """Sync a single rule to the database.

    Use this when saving a rule after editing in the KE workbench.

    Args:
        rule: The rule to sync
        rule_repo: Optional repository instance (creates new if not provided)

    Returns:
        True if successful
    """
    if rule_repo is None:
        rule_repo = RuleRepository()

    yaml_content = _rule_to_yaml(rule)

    rule_repo.save_rule(
        rule_id=rule.rule_id,
        content_yaml=yaml_content,
        source_document_id=rule.source.document_id if rule.source else None,
        source_article=rule.source.article if rule.source else None,
    )

    # Update premise index
    premise_keys = extract_premise_keys(rule.applies_if)
    if premise_keys:
        rule_repo.update_premise_index(rule.rule_id, premise_keys)

    return True


def load_rules_from_db(rule_repo: RuleRepository | None = None) -> dict[str, Rule]:
    """Load all rules from database as Rule objects.

    Args:
        rule_repo: Optional repository instance

    Returns:
        Dict mapping rule_id to Rule objects
    """
    if rule_repo is None:
        rule_repo = RuleRepository()

    loader = RuleLoader()
    rules: dict[str, Rule] = {}

    for record in rule_repo.get_all_rules():
        try:
            # Parse YAML content back to Rule
            data = yaml.safe_load(record.content_yaml)
            rule = loader._parse_rule(data)
            rules[rule.rule_id] = rule
        except Exception as e:
            print(f"Warning: Failed to parse rule {record.rule_id}: {e}")

    return rules


def get_migration_status() -> dict[str, Any]:
    """Get current migration/database status.

    Returns:
        Dict with counts and status information
    """
    init_db()
    rule_repo = RuleRepository()
    verify_repo = VerificationRepository()

    return {
        "rules_count": rule_repo.count_rules(),
        "compiled_rules_count": rule_repo.count_compiled_rules(),
        "verification_stats": verify_repo.get_verification_stats(),
        "reviews_count": verify_repo.count_reviews(),
        "premise_keys": rule_repo.get_all_premise_keys(),
    }


if __name__ == "__main__":
    # Run migration from command line
    import sys

    rules_dir = Path(__file__).parent.parent / "rules"
    clear = "--clear" in sys.argv

    print(f"Migrating rules from {rules_dir}")
    if clear:
        print("Clearing existing data...")

    result = migrate_yaml_rules(rules_dir, clear_existing=clear)

    print(f"\nMigration {'successful' if result['success'] else 'completed with errors'}")
    print(f"  Rules migrated: {result['rules_migrated']}")
    print(f"  Rules updated: {result['rules_updated']}")
    print(f"  Verifications migrated: {result['verifications_migrated']}")
    print(f"  Premise keys indexed: {result['premise_keys_indexed']}")

    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for err in result["errors"]:
            print(f"  - {err}")
