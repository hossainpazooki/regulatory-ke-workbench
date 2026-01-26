"""Verification domain - rule consistency checking and validation."""

from .service import (
    ConsistencyEngine,
    verify_rule,
    # Tier 0 checks
    check_schema_valid,
    check_required_fields,
    check_source_exists,
    check_date_consistency,
    check_id_format,
    check_decision_tree_valid,
    # Tier 1 checks
    check_deontic_alignment,
    check_actor_mentioned,
    check_instrument_mentioned,
    check_keyword_overlap,
    check_negation_consistency,
    check_exception_coverage,
    # Tier 2-4 stubs
    check_semantic_alignment,
    check_entailment,
    check_cross_rule_consistency,
    # Utilities
    compute_summary,
)

__all__ = [
    "ConsistencyEngine",
    "verify_rule",
    # Tier 0 checks
    "check_schema_valid",
    "check_required_fields",
    "check_source_exists",
    "check_date_consistency",
    "check_id_format",
    "check_decision_tree_valid",
    # Tier 1 checks
    "check_deontic_alignment",
    "check_actor_mentioned",
    "check_instrument_mentioned",
    "check_keyword_overlap",
    "check_negation_consistency",
    "check_exception_coverage",
    # Tier 2-4 stubs
    "check_semantic_alignment",
    "check_entailment",
    "check_cross_rule_consistency",
    # Utilities
    "compute_summary",
]
