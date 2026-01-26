"""Semantic consistency engine for rule verification.

Implements verification tiers as defined in docs/semantic_consistency_regulatory_kg.md:
- Tier 0: Schema & Structural Validation (implemented)
- Tier 1: Lexical & Heuristic Analysis (implemented)
- Tier 2: Semantic Similarity (stub)
- Tier 3: NLI-based Entailment (stub)
- Tier 4: Cross-Rule Consistency (stub)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

from backend.rules import (
    Rule,
    ConsistencyBlock,
    ConsistencySummary,
    ConsistencyEvidence,
    ConsistencyStatus,
)


# =============================================================================
# Evidence Factory
# =============================================================================

def _make_evidence(
    tier: int,
    category: str,
    label: str,
    score: float,
    details: str,
    source_span: str | None = None,
    rule_element: str | None = None,
) -> ConsistencyEvidence:
    """Create a ConsistencyEvidence record with current timestamp."""
    return ConsistencyEvidence(
        tier=tier,
        category=category,
        label=label,
        score=score,
        details=details,
        source_span=source_span,
        rule_element=rule_element,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


# =============================================================================
# Tier 0: Schema & Structural Validation
# =============================================================================

def check_schema_valid(rule: Rule) -> ConsistencyEvidence:
    """Check that rule parsed successfully (always pass if we got here)."""
    return _make_evidence(
        tier=0,
        category="schema_valid",
        label="pass",
        score=1.0,
        details="Rule parses against DSL schema",
    )


def check_required_fields(rule: Rule) -> ConsistencyEvidence:
    """Check that required fields are present."""
    missing = []

    if not rule.rule_id:
        missing.append("rule_id")
    if not rule.source:
        missing.append("source")

    if missing:
        return _make_evidence(
            tier=0,
            category="required_fields",
            label="fail",
            score=0.0,
            details=f"Missing required fields: {', '.join(missing)}",
            rule_element="root",
        )

    return _make_evidence(
        tier=0,
        category="required_fields",
        label="pass",
        score=1.0,
        details="All required fields present (rule_id, source)",
    )


def check_source_exists(rule: Rule, document_registry: dict[str, str] | None = None) -> ConsistencyEvidence:
    """Check that source document reference is valid.

    Args:
        rule: The rule to check.
        document_registry: Optional mapping of document_id to document path/URL.
    """
    if not rule.source:
        return _make_evidence(
            tier=0,
            category="source_exists",
            label="fail",
            score=0.0,
            details="No source reference provided",
            rule_element="source",
        )

    # If we have a registry, check if document exists
    if document_registry:
        if rule.source.document_id not in document_registry:
            return _make_evidence(
                tier=0,
                category="source_exists",
                label="warning",
                score=0.5,
                details=f"Document '{rule.source.document_id}' not found in registry",
                rule_element="source.document_id",
            )

    # Basic validation: document_id should be non-empty
    if not rule.source.document_id:
        return _make_evidence(
            tier=0,
            category="source_exists",
            label="fail",
            score=0.0,
            details="Empty document_id in source reference",
            rule_element="source.document_id",
        )

    return _make_evidence(
        tier=0,
        category="source_exists",
        label="pass",
        score=1.0,
        details=f"Source reference valid: {rule.source.document_id}",
    )


def check_date_consistency(rule: Rule) -> ConsistencyEvidence:
    """Check that effective_from <= effective_to if both are present."""
    if rule.effective_from and rule.effective_to:
        if rule.effective_from > rule.effective_to:
            return _make_evidence(
                tier=0,
                category="date_consistency",
                label="fail",
                score=0.0,
                details=f"effective_from ({rule.effective_from}) > effective_to ({rule.effective_to})",
                rule_element="effective_from, effective_to",
            )

    return _make_evidence(
        tier=0,
        category="date_consistency",
        label="pass",
        score=1.0,
        details="Date consistency check passed",
    )


def check_id_format(rule: Rule) -> ConsistencyEvidence:
    """Check that rule_id follows naming convention (snake_case, prefixed)."""
    rule_id = rule.rule_id

    # Pattern: lowercase letters, numbers, underscores; should have meaningful prefix
    pattern = r"^[a-z][a-z0-9_]*$"

    if not re.match(pattern, rule_id):
        return _make_evidence(
            tier=0,
            category="id_format",
            label="warning",
            score=0.7,
            details=f"rule_id '{rule_id}' doesn't follow snake_case convention",
            rule_element="rule_id",
        )

    # Check for meaningful prefix (at least one underscore suggesting structure)
    if "_" not in rule_id:
        return _make_evidence(
            tier=0,
            category="id_format",
            label="warning",
            score=0.8,
            details=f"rule_id '{rule_id}' lacks structured prefix (e.g., 'mica_art36_...')",
            rule_element="rule_id",
        )

    return _make_evidence(
        tier=0,
        category="id_format",
        label="pass",
        score=1.0,
        details=f"rule_id '{rule_id}' follows naming convention",
    )


def check_decision_tree_valid(rule: Rule) -> ConsistencyEvidence:
    """Check that decision tree has valid structure."""
    if not rule.decision_tree:
        return _make_evidence(
            tier=0,
            category="decision_tree_valid",
            label="warning",
            score=0.5,
            details="No decision tree defined",
            rule_element="decision_tree",
        )

    # Check for result in leaf nodes
    def has_result(node) -> bool:
        if hasattr(node, "result"):
            return bool(node.result)
        if hasattr(node, "true_branch") or hasattr(node, "false_branch"):
            true_ok = not node.true_branch or has_result(node.true_branch)
            false_ok = not node.false_branch or has_result(node.false_branch)
            return true_ok and false_ok
        return True

    if not has_result(rule.decision_tree):
        return _make_evidence(
            tier=0,
            category="decision_tree_valid",
            label="warning",
            score=0.7,
            details="Decision tree has branches without result values",
            rule_element="decision_tree",
        )

    return _make_evidence(
        tier=0,
        category="decision_tree_valid",
        label="pass",
        score=1.0,
        details="Decision tree structure valid",
    )


# =============================================================================
# Tier 1: Lexical & Heuristic Analysis
# =============================================================================

# Deontic verb patterns
DEONTIC_OBLIGATION = re.compile(r"\b(shall|must|required to|obliged to|has to|have to)\b", re.IGNORECASE)
DEONTIC_PERMISSION = re.compile(r"\b(may|can|permitted to|allowed to|entitled to)\b", re.IGNORECASE)
DEONTIC_PROHIBITION = re.compile(r"\b(shall not|must not|may not|prohibited|forbidden)\b", re.IGNORECASE)

# Actor type keywords
ACTOR_KEYWORDS = {
    "issuer": ["issuer", "issuers", "issuing"],
    "offeror": ["offeror", "offerors", "offering party"],
    "trading_platform": ["trading platform", "trading venue", "exchange"],
    "custodian": ["custodian", "custodians", "custody provider"],
    "investor": ["investor", "investors", "holder", "holders"],
    "competent_authority": ["competent authority", "authorities", "regulator"],
}

# Instrument type keywords
INSTRUMENT_KEYWORDS = {
    "art": ["asset-referenced token", "art", "asset referenced"],
    "emt": ["e-money token", "emt", "electronic money token"],
    "stablecoin": ["stablecoin", "stable coin", "stable-coin"],
    "utility_token": ["utility token", "utility tokens"],
    "security_token": ["security token", "security tokens"],
    "nft": ["nft", "non-fungible token"],
}


def check_deontic_alignment(rule: Rule, source_text: str | None = None) -> ConsistencyEvidence:
    """Check that deontic verbs in source match rule modality.

    Args:
        rule: The rule to check.
        source_text: The source legal text. If None, returns warning.
    """
    if not source_text:
        return _make_evidence(
            tier=1,
            category="deontic_alignment",
            label="warning",
            score=0.5,
            details="No source text provided for deontic analysis",
        )

    source_lower = source_text.lower()

    # Detect deontic modality in source
    has_obligation = bool(DEONTIC_OBLIGATION.search(source_text))
    has_permission = bool(DEONTIC_PERMISSION.search(source_text))
    has_prohibition = bool(DEONTIC_PROHIBITION.search(source_text))

    # Detect rule modality from decision tree results
    rule_results = _extract_results(rule)
    rule_results_lower = " ".join(rule_results).lower()

    rule_has_obligation = any(
        word in rule_results_lower
        for word in ["required", "must", "obligation", "mandatory"]
    )
    rule_has_permission = any(
        word in rule_results_lower
        for word in ["permitted", "allowed", "may", "can"]
    )
    rule_has_prohibition = any(
        word in rule_results_lower
        for word in ["prohibited", "forbidden", "not allowed", "banned"]
    )

    # Check alignment
    issues = []
    if has_obligation and not rule_has_obligation:
        issues.append("Source has obligation language but rule doesn't encode obligation")
    if has_prohibition and not rule_has_prohibition:
        issues.append("Source has prohibition language but rule doesn't encode prohibition")

    if issues:
        return _make_evidence(
            tier=1,
            category="deontic_alignment",
            label="warning",
            score=0.6,
            details="; ".join(issues),
            source_span=_find_deontic_span(source_text),
            rule_element="decision_tree",
        )

    return _make_evidence(
        tier=1,
        category="deontic_alignment",
        label="pass",
        score=0.9,
        details="Deontic modality in source aligns with rule encoding",
    )


def check_actor_mentioned(rule: Rule, source_text: str | None = None) -> ConsistencyEvidence:
    """Check that actor types in rule appear in source text."""
    if not source_text:
        return _make_evidence(
            tier=1,
            category="actor_mentioned",
            label="warning",
            score=0.5,
            details="No source text provided for actor analysis",
        )

    source_lower = source_text.lower()

    # Extract actors from rule conditions
    rule_actors = _extract_actors_from_rule(rule)

    if not rule_actors:
        return _make_evidence(
            tier=1,
            category="actor_mentioned",
            label="pass",
            score=1.0,
            details="No specific actor types in rule to verify",
        )

    # Check which actors are mentioned
    unmentioned = []
    for actor in rule_actors:
        keywords = ACTOR_KEYWORDS.get(actor.lower(), [actor.lower()])
        if not any(kw in source_lower for kw in keywords):
            unmentioned.append(actor)

    if unmentioned:
        return _make_evidence(
            tier=1,
            category="actor_mentioned",
            label="warning",
            score=0.7,
            details=f"Actor types in rule not found in source: {', '.join(unmentioned)}",
            rule_element="applies_if",
        )

    return _make_evidence(
        tier=1,
        category="actor_mentioned",
        label="pass",
        score=0.9,
        details="All actor types in rule found in source text",
    )


def check_instrument_mentioned(rule: Rule, source_text: str | None = None) -> ConsistencyEvidence:
    """Check that instrument types in rule appear in source text."""
    if not source_text:
        return _make_evidence(
            tier=1,
            category="instrument_mentioned",
            label="warning",
            score=0.5,
            details="No source text provided for instrument analysis",
        )

    source_lower = source_text.lower()

    # Extract instruments from rule
    rule_instruments = _extract_instruments_from_rule(rule)

    if not rule_instruments:
        return _make_evidence(
            tier=1,
            category="instrument_mentioned",
            label="pass",
            score=1.0,
            details="No specific instrument types in rule to verify",
        )

    # Check which instruments are mentioned
    unmentioned = []
    for instrument in rule_instruments:
        keywords = INSTRUMENT_KEYWORDS.get(instrument.lower(), [instrument.lower()])
        if not any(kw in source_lower for kw in keywords):
            unmentioned.append(instrument)

    if unmentioned:
        return _make_evidence(
            tier=1,
            category="instrument_mentioned",
            label="warning",
            score=0.7,
            details=f"Instrument types in rule not found in source: {', '.join(unmentioned)}",
            rule_element="applies_if",
        )

    return _make_evidence(
        tier=1,
        category="instrument_mentioned",
        label="pass",
        score=0.9,
        details="All instrument types in rule found in source text",
    )


def check_keyword_overlap(rule: Rule, source_text: str | None = None) -> ConsistencyEvidence:
    """Check that key terms from rule appear in source."""
    if not source_text:
        return _make_evidence(
            tier=1,
            category="keyword_overlap",
            label="warning",
            score=0.5,
            details="No source text provided for keyword analysis",
        )

    source_lower = source_text.lower()

    # Extract keywords from rule
    rule_keywords = _extract_keywords_from_rule(rule)

    if not rule_keywords:
        return _make_evidence(
            tier=1,
            category="keyword_overlap",
            label="pass",
            score=1.0,
            details="No significant keywords extracted from rule",
        )

    # Count overlapping keywords
    found = [kw for kw in rule_keywords if kw.lower() in source_lower]
    overlap_ratio = len(found) / len(rule_keywords) if rule_keywords else 0

    if overlap_ratio < 0.3:
        return _make_evidence(
            tier=1,
            category="keyword_overlap",
            label="warning",
            score=overlap_ratio,
            details=f"Low keyword overlap ({len(found)}/{len(rule_keywords)}): missing {set(rule_keywords) - set(found)}",
            rule_element="applies_if, decision_tree",
        )

    return _make_evidence(
        tier=1,
        category="keyword_overlap",
        label="pass",
        score=min(1.0, overlap_ratio + 0.2),
        details=f"Good keyword overlap ({len(found)}/{len(rule_keywords)})",
    )


def check_negation_consistency(rule: Rule, source_text: str | None = None) -> ConsistencyEvidence:
    """Check that negations in source are reflected in rule logic."""
    if not source_text:
        return _make_evidence(
            tier=1,
            category="negation_consistency",
            label="warning",
            score=0.5,
            details="No source text provided for negation analysis",
        )

    # Detect negation patterns in source
    negation_patterns = [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bwithout\b",
        r"\bexcept\b",
        r"\bunless\b",
        r"\bexcluding\b",
    ]

    source_has_negation = any(
        re.search(pattern, source_text, re.IGNORECASE)
        for pattern in negation_patterns
    )

    # Check if rule has negation in conditions
    rule_has_negation = _rule_has_negation(rule)

    if source_has_negation and not rule_has_negation:
        # Find the negation span
        negation_span = None
        for pattern in negation_patterns:
            match = re.search(pattern + r".{0,50}", source_text, re.IGNORECASE)
            if match:
                negation_span = match.group(0)
                break

        return _make_evidence(
            tier=1,
            category="negation_consistency",
            label="warning",
            score=0.6,
            details="Source contains negation that may not be reflected in rule",
            source_span=negation_span,
            rule_element="applies_if",
        )

    return _make_evidence(
        tier=1,
        category="negation_consistency",
        label="pass",
        score=0.9,
        details="Negation patterns consistent between source and rule",
    )


def check_exception_coverage(rule: Rule, source_text: str | None = None) -> ConsistencyEvidence:
    """Check that exceptions in source have corresponding branches."""
    if not source_text:
        return _make_evidence(
            tier=1,
            category="exception_coverage",
            label="warning",
            score=0.5,
            details="No source text provided for exception analysis",
        )

    # Detect exception language in source
    exception_patterns = [
        (r"\bexcept\s+(?:where|when|if|that)\b", "except"),
        (r"\bunless\b", "unless"),
        (r"\bprovided\s+that\b", "provided that"),
        (r"\bnotwithstanding\b", "notwithstanding"),
        (r"\bsubject\s+to\b", "subject to"),
        (r"\bother\s+than\b", "other than"),
    ]

    found_exceptions = []
    for pattern, name in exception_patterns:
        if re.search(pattern, source_text, re.IGNORECASE):
            found_exceptions.append(name)

    if not found_exceptions:
        return _make_evidence(
            tier=1,
            category="exception_coverage",
            label="pass",
            score=1.0,
            details="No exception language detected in source",
        )

    # Check if rule has branching logic
    has_branches = _count_branches(rule) > 1

    if not has_branches:
        return _make_evidence(
            tier=1,
            category="exception_coverage",
            label="warning",
            score=0.5,
            details=f"Source has exception language ({', '.join(found_exceptions)}) but rule lacks branching",
            rule_element="decision_tree",
        )

    return _make_evidence(
        tier=1,
        category="exception_coverage",
        label="pass",
        score=0.85,
        details=f"Rule has branches that may cover exceptions ({', '.join(found_exceptions)})",
    )


# =============================================================================
# Tier 2-4: Stubs (require ML dependencies)
# =============================================================================

def check_semantic_alignment(rule: Rule, source_text: str | None = None) -> ConsistencyEvidence:
    """Stub: Check semantic similarity between rule logic and source.

    Requires: sentence-transformers
    """
    return _make_evidence(
        tier=2,
        category="semantic_alignment",
        label="warning",
        score=0.0,
        details="Tier 2 semantic checks not implemented (requires ML dependencies)",
    )


def check_entailment(rule: Rule, source_text: str | None = None) -> ConsistencyEvidence:
    """Stub: Check NLI entailment between source and rule conclusion.

    Requires: NLI model (e.g., roberta-large-mnli)
    """
    return _make_evidence(
        tier=3,
        category="entailment",
        label="warning",
        score=0.0,
        details="Tier 3 NLI checks not implemented (requires NLI model)",
    )


def check_cross_rule_consistency(
    rule: Rule,
    related_rules: list[Rule] | None = None
) -> ConsistencyEvidence:
    """Stub: Check consistency with related rules.

    Requires: Rule graph traversal, potential contradiction detection
    """
    return _make_evidence(
        tier=4,
        category="no_contradiction",
        label="warning",
        score=0.0,
        details="Tier 4 cross-rule checks not implemented",
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_results(rule: Rule) -> list[str]:
    """Extract all result strings from decision tree."""
    results = []

    def traverse(node):
        if node is None:
            return
        if hasattr(node, "result") and node.result:
            results.append(node.result)
        if hasattr(node, "true_branch"):
            traverse(node.true_branch)
        if hasattr(node, "false_branch"):
            traverse(node.false_branch)

    if rule.decision_tree:
        traverse(rule.decision_tree)

    return results


def _find_deontic_span(text: str, max_len: int = 60) -> str | None:
    """Find a span containing deontic verbs."""
    for pattern in [DEONTIC_OBLIGATION, DEONTIC_PROHIBITION, DEONTIC_PERMISSION]:
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 40)
            return text[start:end].strip()
    return None


def _extract_actors_from_rule(rule: Rule) -> list[str]:
    """Extract actor type references from rule conditions."""
    actors = []

    def check_condition(cond):
        if hasattr(cond, "field"):
            field_lower = cond.field.lower()
            if "actor" in field_lower or "type" in field_lower:
                if cond.value:
                    if isinstance(cond.value, list):
                        actors.extend(str(v) for v in cond.value)
                    else:
                        actors.append(str(cond.value))
        if hasattr(cond, "all") and cond.all:
            for c in cond.all:
                check_condition(c)
        if hasattr(cond, "any") and cond.any:
            for c in cond.any:
                check_condition(c)

    if rule.applies_if:
        check_condition(rule.applies_if)

    return actors


def _extract_instruments_from_rule(rule: Rule) -> list[str]:
    """Extract instrument type references from rule conditions."""
    instruments = []

    def check_condition(cond):
        if hasattr(cond, "field"):
            field_lower = cond.field.lower()
            if "instrument" in field_lower or "token" in field_lower:
                if cond.value:
                    if isinstance(cond.value, list):
                        instruments.extend(str(v) for v in cond.value)
                    else:
                        instruments.append(str(cond.value))
        if hasattr(cond, "all") and cond.all:
            for c in cond.all:
                check_condition(c)
        if hasattr(cond, "any") and cond.any:
            for c in cond.any:
                check_condition(c)

    if rule.applies_if:
        check_condition(rule.applies_if)

    return instruments


def _extract_keywords_from_rule(rule: Rule) -> list[str]:
    """Extract significant keywords from rule."""
    keywords = set()

    # From rule description
    if rule.description:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", rule.description)
        keywords.update(w.lower() for w in words)

    # From tags
    keywords.update(tag.lower() for tag in rule.tags)

    # From results
    for result in _extract_results(rule):
        words = re.findall(r"\b[a-zA-Z]{4,}\b", result)
        keywords.update(w.lower() for w in words)

    # Remove common words
    stopwords = {"this", "that", "with", "from", "have", "been", "were", "being"}
    keywords -= stopwords

    return list(keywords)


def _rule_has_negation(rule: Rule) -> bool:
    """Check if rule conditions contain negation operators."""
    def check_condition(cond) -> bool:
        if hasattr(cond, "operator"):
            if cond.operator in ("!=", "not_in", "not"):
                return True
        if hasattr(cond, "all") and cond.all:
            if any(check_condition(c) for c in cond.all):
                return True
        if hasattr(cond, "any") and cond.any:
            if any(check_condition(c) for c in cond.any):
                return True
        return False

    if rule.applies_if:
        return check_condition(rule.applies_if)

    # Also check decision tree conditions
    def check_tree(node) -> bool:
        if node is None:
            return False
        if hasattr(node, "condition") and node.condition:
            if hasattr(node.condition, "operator"):
                if node.condition.operator in ("!=", "not_in", "not"):
                    return True
        if hasattr(node, "true_branch") and check_tree(node.true_branch):
            return True
        if hasattr(node, "false_branch") and check_tree(node.false_branch):
            return True
        return False

    if rule.decision_tree:
        return check_tree(rule.decision_tree)

    return False


def _count_branches(rule: Rule) -> int:
    """Count the number of decision branches in rule."""
    count = 0

    def traverse(node):
        nonlocal count
        if node is None:
            return
        if hasattr(node, "result"):
            count += 1
        if hasattr(node, "true_branch"):
            traverse(node.true_branch)
        if hasattr(node, "false_branch"):
            traverse(node.false_branch)

    if rule.decision_tree:
        traverse(rule.decision_tree)

    return count


# =============================================================================
# Summary Computation
# =============================================================================

def compute_summary(evidence: list[ConsistencyEvidence]) -> ConsistencySummary:
    """Compute summary from evidence list.

    See docs/semantic_consistency_regulatory_kg.md for algorithm.
    """
    if not evidence:
        return ConsistencySummary(
            status=ConsistencyStatus.UNVERIFIED,
            confidence=0.0,
            last_verified=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            verified_by="system",
        )

    # Status determination
    if any(e.label == "fail" for e in evidence):
        status = ConsistencyStatus.INCONSISTENT
    elif any(e.label == "warning" for e in evidence):
        status = ConsistencyStatus.NEEDS_REVIEW
    else:
        status = ConsistencyStatus.VERIFIED

    # Confidence is weighted average of scores
    tier_weights = {0: 1.0, 1: 0.8, 2: 0.9, 3: 0.95, 4: 0.7}
    weighted_sum = sum(
        e.score * tier_weights.get(e.tier, 0.5)
        for e in evidence
    )
    total_weight = sum(tier_weights.get(e.tier, 0.5) for e in evidence)
    confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

    return ConsistencySummary(
        status=status,
        confidence=round(confidence, 4),
        last_verified=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        verified_by="system",
    )


# =============================================================================
# Main Consistency Engine
# =============================================================================

class ConsistencyEngine:
    """Engine for verifying rule consistency across all tiers."""

    def __init__(
        self,
        document_registry: dict[str, str] | None = None,
        retriever=None,
    ):
        """Initialize the consistency engine.

        Args:
            document_registry: Mapping of document_id to document path/URL.
            retriever: Optional RAG retriever for fetching source text.
        """
        self.document_registry = document_registry or {}
        self.retriever = retriever

        # Register tier 0 checks
        self._tier0_checks: list[Callable] = [
            check_schema_valid,
            lambda r: check_required_fields(r),
            lambda r: check_source_exists(r, self.document_registry),
            check_date_consistency,
            check_id_format,
            check_decision_tree_valid,
        ]

        # Register tier 1 checks (require source text)
        self._tier1_checks: list[Callable] = [
            check_deontic_alignment,
            check_actor_mentioned,
            check_instrument_mentioned,
            check_keyword_overlap,
            check_negation_consistency,
            check_exception_coverage,
        ]

    def verify_rule(
        self,
        rule: Rule,
        source_text: str | None = None,
        tiers: list[int] | None = None,
    ) -> ConsistencyBlock:
        """Run consistency checks on a rule.

        Args:
            rule: The rule to verify.
            source_text: The source legal text. If None and retriever available,
                will attempt to fetch.
            tiers: Which tiers to run (default: [0, 1]).

        Returns:
            ConsistencyBlock with summary and evidence.
        """
        if tiers is None:
            tiers = [0, 1]

        evidence = []

        # Try to get source text if not provided
        if source_text is None and self.retriever is not None and rule.source:
            source_text = self._fetch_source_text(rule)

        # Run tier 0 checks
        if 0 in tiers:
            for check in self._tier0_checks:
                try:
                    evidence.append(check(rule))
                except Exception as e:
                    evidence.append(_make_evidence(
                        tier=0,
                        category="check_error",
                        label="fail",
                        score=0.0,
                        details=f"Check failed with error: {e}",
                    ))

        # Run tier 1 checks
        if 1 in tiers:
            for check in self._tier1_checks:
                try:
                    evidence.append(check(rule, source_text))
                except Exception as e:
                    evidence.append(_make_evidence(
                        tier=1,
                        category="check_error",
                        label="fail",
                        score=0.0,
                        details=f"Check failed with error: {e}",
                    ))

        # Run tier 2+ stubs if requested
        if 2 in tiers:
            evidence.append(check_semantic_alignment(rule, source_text))
        if 3 in tiers:
            evidence.append(check_entailment(rule, source_text))
        if 4 in tiers:
            evidence.append(check_cross_rule_consistency(rule))

        # Compute summary
        summary = compute_summary(evidence)

        return ConsistencyBlock(summary=summary, evidence=evidence)

    def _fetch_source_text(self, rule: Rule) -> str | None:
        """Fetch source text using retriever."""
        if not self.retriever or not rule.source:
            return None

        # Build search query from source reference
        query_parts = []
        if rule.source.document_id:
            query_parts.append(rule.source.document_id)
        if rule.source.article:
            query_parts.append(f"Article {rule.source.article}")

        if not query_parts:
            return None

        query = " ".join(query_parts)
        results = self.retriever.search(query, top_k=3)

        if results:
            # Concatenate top results
            return " ".join(r.text for r in results)

        return None


def verify_rule(
    rule: Rule,
    source_text: str | None = None,
    document_registry: dict[str, str] | None = None,
) -> ConsistencyBlock:
    """Convenience function to verify a single rule.

    Args:
        rule: The rule to verify.
        source_text: The source legal text.
        document_registry: Optional document registry.

    Returns:
        ConsistencyBlock with verification results.
    """
    engine = ConsistencyEngine(document_registry=document_registry)
    return engine.verify_rule(rule, source_text)
