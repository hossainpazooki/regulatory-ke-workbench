"""
Database integration helpers for KE Workbench frontend.

Provides functions for the Streamlit dashboard to interact with:
- Rule persistence (load from DB, sync to DB)
- Compilation (compile rules to IR)
- Verification result persistence
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.storage import (
    init_db,
    RuleRepository,
    VerificationRepository,
    migrate_yaml_rules,
    load_rules_from_db,
    sync_rule_to_db,
    get_migration_status,
)
from backend.storage.retrieval.compiler import RuleCompiler, PremiseIndexBuilder
from backend.storage.retrieval.compiler.ir import RuleIR
from backend.storage.retrieval.compiler.optimizer import optimize_rule
from backend.storage.retrieval.runtime import RuleRuntime, IRCache
from backend.rules.loader import Rule
from backend.rules.schema import ConsistencyBlock


class DatabaseState:
    """Manages database state for the frontend."""

    def __init__(self):
        self._initialized = False
        self._rule_repo: RuleRepository | None = None
        self._verify_repo: VerificationRepository | None = None
        self._runtime: RuleRuntime | None = None
        self._premise_index: PremiseIndexBuilder | None = None
        self._ir_cache: IRCache | None = None

    def initialize(self) -> None:
        """Initialize database connection and repositories."""
        if not self._initialized:
            init_db()
            self._rule_repo = RuleRepository()
            self._verify_repo = VerificationRepository()
            self._ir_cache = IRCache()
            self._premise_index = PremiseIndexBuilder()
            self._runtime = RuleRuntime(
                cache=self._ir_cache,
                premise_index=self._premise_index,
            )
            self._initialized = True

    @property
    def rule_repo(self) -> RuleRepository:
        """Get rule repository."""
        self.initialize()
        return self._rule_repo

    @property
    def verify_repo(self) -> VerificationRepository:
        """Get verification repository."""
        self.initialize()
        return self._verify_repo

    @property
    def runtime(self) -> RuleRuntime:
        """Get rule runtime."""
        self.initialize()
        return self._runtime


# Global state instance
_db_state: DatabaseState | None = None


def get_db_state() -> DatabaseState:
    """Get or create the global database state."""
    global _db_state
    if _db_state is None:
        _db_state = DatabaseState()
    return _db_state


def migrate_rules_to_db(clear_existing: bool = False) -> dict[str, Any]:
    """Migrate YAML rules from disk to database.

    Args:
        clear_existing: If True, clears existing data before migration

    Returns:
        Migration result dict
    """
    rules_dir = Path(__file__).parent.parent.parent / "backend" / "rules"
    return migrate_yaml_rules(rules_dir, clear_existing=clear_existing)


def get_database_status() -> dict[str, Any]:
    """Get current database status.

    Returns:
        Dict with rules count, compiled count, verification stats, etc.
    """
    return get_migration_status()


def load_rules() -> dict[str, Rule]:
    """Load all rules from database as Rule objects.

    Returns:
        Dict mapping rule_id to Rule objects
    """
    db = get_db_state()
    return load_rules_from_db(db.rule_repo)


def sync_rule(rule: Rule) -> bool:
    """Sync a rule to the database.

    Args:
        rule: The rule to sync

    Returns:
        True if successful
    """
    return sync_rule_to_db(rule, get_db_state().rule_repo)


def compile_rule(rule_id: str, optimize: bool = True) -> dict[str, Any]:
    """Compile a single rule to IR.

    Args:
        rule_id: The rule to compile
        optimize: Whether to apply optimizations

    Returns:
        Dict with compilation result
    """
    db = get_db_state()
    record = db.rule_repo.get_rule(rule_id)

    if record is None:
        return {"success": False, "error": f"Rule not found: {rule_id}"}

    try:
        # Load rules to get parsed Rule object
        rules = load_rules()
        rule = rules.get(rule_id)

        if rule is None:
            return {"success": False, "error": f"Failed to parse rule: {rule_id}"}

        # Compile
        compiler = RuleCompiler()
        ir = compiler.compile(rule, record.content_yaml)

        if optimize:
            ir = optimize_rule(ir)

        # Store
        db.rule_repo.update_rule_ir(rule_id, ir.to_json())
        db.rule_repo.update_premise_index(rule_id, ir.premise_keys)

        # Update cache
        db.runtime._cache.put(rule_id, ir)
        db.runtime._premise_index.add_rule(ir)

        return {
            "success": True,
            "rule_id": rule_id,
            "premise_keys": ir.premise_keys,
            "applicability_checks": len(ir.applicability_checks),
            "decision_table_size": len(ir.decision_table),
            "compiled_at": ir.compiled_at,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def compile_all_rules(optimize: bool = True) -> dict[str, Any]:
    """Compile all rules to IR.

    Args:
        optimize: Whether to apply optimizations

    Returns:
        Dict with compilation results
    """
    db = get_db_state()
    rules = load_rules()

    compiler = RuleCompiler()
    compiled = 0
    errors = []

    for rule_id, rule in rules.items():
        try:
            record = db.rule_repo.get_rule(rule_id)
            yaml_content = record.content_yaml if record else None

            ir = compiler.compile(rule, yaml_content)
            if optimize:
                ir = optimize_rule(ir)

            db.rule_repo.update_rule_ir(rule_id, ir.to_json())
            db.rule_repo.update_premise_index(rule_id, ir.premise_keys)

            db.runtime._cache.put(rule_id, ir)
            db.runtime._premise_index.add_rule(ir)

            compiled += 1
        except Exception as e:
            errors.append({"rule_id": rule_id, "error": str(e)})

    return {
        "total": len(rules),
        "compiled": compiled,
        "failed": len(errors),
        "errors": errors,
    }


def save_verification_result(
    rule_id: str,
    consistency: ConsistencyBlock,
) -> bool:
    """Save a verification result to the database.

    Args:
        rule_id: The rule ID
        consistency: The consistency block from verification

    Returns:
        True if successful
    """
    db = get_db_state()

    evidence_list = [
        {
            "tier": ev.tier,
            "category": ev.category,
            "label": ev.label,
            "score": ev.score,
            "details": ev.details,
            "source_span": ev.source_span,
            "rule_element": ev.rule_element,
        }
        for ev in consistency.evidence
    ]

    db.verify_repo.save_verification_result(
        rule_id=rule_id,
        status=consistency.summary.status.value,
        confidence=consistency.summary.confidence,
        verified_by=consistency.summary.verified_by,
        notes=consistency.summary.notes,
        evidence=evidence_list,
    )

    return True


def load_verification_result(rule_id: str) -> dict[str, Any] | None:
    """Load a verification result from the database.

    Args:
        rule_id: The rule ID

    Returns:
        Dict with verification data or None
    """
    db = get_db_state()
    result, evidence = db.verify_repo.get_verification_result(rule_id)

    if result is None:
        return None

    return {
        "rule_id": result.rule_id,
        "status": result.status,
        "confidence": result.confidence,
        "verified_at": result.verified_at,
        "verified_by": result.verified_by,
        "notes": result.notes,
        "evidence": [
            {
                "tier": ev.tier,
                "category": ev.category,
                "label": ev.label,
                "score": ev.score,
                "details": ev.details,
            }
            for ev in evidence
        ],
    }


def load_all_verification_results() -> dict[str, dict[str, Any]]:
    """Load all verification results from database.

    Returns:
        Dict mapping rule_id to verification data
    """
    db = get_db_state()
    all_results = db.verify_repo.get_all_verification_results()

    return {
        rule_id: {
            "rule_id": result.rule_id,
            "status": result.status,
            "confidence": result.confidence,
            "verified_at": result.verified_at,
            "verified_by": result.verified_by,
            "evidence_count": len(evidence),
        }
        for rule_id, (result, evidence) in all_results.items()
    }


def save_human_review(
    rule_id: str,
    reviewer_id: str,
    decision: str,
    notes: str | None = None,
) -> bool:
    """Save a human review to the database.

    Args:
        rule_id: The rule ID
        reviewer_id: ID of the reviewer
        decision: Review decision (consistent, inconsistent, unknown)
        notes: Optional notes

    Returns:
        True if successful
    """
    db = get_db_state()
    db.verify_repo.save_review(
        rule_id=rule_id,
        reviewer_id=reviewer_id,
        decision=decision,
        notes=notes,
    )
    return True


def get_rule_reviews(rule_id: str) -> list[dict[str, Any]]:
    """Get all reviews for a rule.

    Args:
        rule_id: The rule ID

    Returns:
        List of review dicts
    """
    db = get_db_state()
    reviews = db.verify_repo.get_reviews_for_rule(rule_id)

    return [
        {
            "reviewer_id": r.reviewer_id,
            "decision": r.decision,
            "notes": r.notes,
            "created_at": r.created_at,
        }
        for r in reviews
    ]


def get_compilation_status(rule_id: str) -> dict[str, Any] | None:
    """Get compilation status for a rule.

    Args:
        rule_id: The rule ID

    Returns:
        Dict with compilation info or None
    """
    db = get_db_state()
    record = db.rule_repo.get_rule(rule_id)

    if record is None:
        return None

    if record.rule_ir is None:
        return {
            "compiled": False,
            "compiled_at": None,
        }

    try:
        ir = RuleIR.from_json(record.rule_ir)
        return {
            "compiled": True,
            "compiled_at": record.compiled_at,
            "premise_keys": ir.premise_keys,
            "applicability_checks": len(ir.applicability_checks),
            "decision_table_size": len(ir.decision_table),
        }
    except Exception:
        return {
            "compiled": False,
            "error": "Failed to parse IR",
        }


def get_cache_stats() -> dict[str, Any]:
    """Get IR cache statistics.

    Returns:
        Dict with cache stats
    """
    db = get_db_state()
    return db.runtime._cache.get_stats()


def get_index_stats() -> dict[str, Any]:
    """Get premise index statistics.

    Returns:
        Dict with index stats
    """
    db = get_db_state()
    return db.runtime._premise_index.get_stats()
