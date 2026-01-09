"""Tests for analytics module (error patterns and drift detection)."""

from __future__ import annotations

import pytest
from datetime import date

from backend.rule_service.app.services.loader import (
    Rule,
    RuleLoader,
    SourceRef,
    DecisionLeaf,
)
from backend.rule_service.app.services.schema import (
    ConsistencyBlock,
    ConsistencySummary,
    ConsistencyEvidence,
    ConsistencyStatus,
)
from backend.analytics_service.app.services import (
    ErrorPatternAnalyzer,
    ErrorPattern,
    CategoryStats,
    DriftDetector,
    DriftReport,
    DriftMetrics,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def rules_with_consistency() -> list[Rule]:
    """Create a set of rules with various consistency states."""
    rules = []

    # Rule 1: Verified, all passing
    rules.append(Rule(
        rule_id="rule_verified",
        source=SourceRef(document_id="test"),
        decision_tree=DecisionLeaf(result="test"),
        consistency=ConsistencyBlock(
            summary=ConsistencySummary(
                status=ConsistencyStatus.VERIFIED,
                confidence=0.95,
                verified_by="system",
            ),
            evidence=[
                ConsistencyEvidence(
                    tier=0, category="schema_valid", label="pass", score=1.0, details="OK"
                ),
                ConsistencyEvidence(
                    tier=0, category="required_fields", label="pass", score=1.0, details="OK"
                ),
                ConsistencyEvidence(
                    tier=1, category="deontic_alignment", label="pass", score=0.9, details="OK"
                ),
            ],
        ),
    ))

    # Rule 2: Needs review, some warnings
    rules.append(Rule(
        rule_id="rule_needs_review",
        source=SourceRef(document_id="test"),
        decision_tree=DecisionLeaf(result="test"),
        consistency=ConsistencyBlock(
            summary=ConsistencySummary(
                status=ConsistencyStatus.NEEDS_REVIEW,
                confidence=0.7,
                verified_by="system",
            ),
            evidence=[
                ConsistencyEvidence(
                    tier=0, category="schema_valid", label="pass", score=1.0, details="OK"
                ),
                ConsistencyEvidence(
                    tier=1, category="deontic_alignment", label="warning", score=0.6, details="Mismatch"
                ),
                ConsistencyEvidence(
                    tier=1, category="keyword_overlap", label="warning", score=0.5, details="Low overlap"
                ),
            ],
        ),
    ))

    # Rule 3: Inconsistent, with failures
    rules.append(Rule(
        rule_id="rule_inconsistent",
        source=SourceRef(document_id="test"),
        decision_tree=DecisionLeaf(result="test"),
        consistency=ConsistencyBlock(
            summary=ConsistencySummary(
                status=ConsistencyStatus.INCONSISTENT,
                confidence=0.3,
                verified_by="system",
            ),
            evidence=[
                ConsistencyEvidence(
                    tier=0, category="schema_valid", label="pass", score=1.0, details="OK"
                ),
                ConsistencyEvidence(
                    tier=0, category="required_fields", label="fail", score=0.0, details="Missing source"
                ),
                ConsistencyEvidence(
                    tier=1, category="deontic_alignment", label="fail", score=0.2, details="Major mismatch"
                ),
            ],
        ),
    ))

    # Rule 4: Unverified (no consistency block)
    rules.append(Rule(
        rule_id="rule_unverified",
        source=SourceRef(document_id="test"),
        decision_tree=DecisionLeaf(result="test"),
    ))

    # Rule 5: Another warning case
    rules.append(Rule(
        rule_id="rule_warnings_2",
        source=SourceRef(document_id="test"),
        decision_tree=DecisionLeaf(result="test"),
        consistency=ConsistencyBlock(
            summary=ConsistencySummary(
                status=ConsistencyStatus.NEEDS_REVIEW,
                confidence=0.65,
                verified_by="human:reviewer1",
            ),
            evidence=[
                ConsistencyEvidence(
                    tier=0, category="schema_valid", label="pass", score=1.0, details="OK"
                ),
                ConsistencyEvidence(
                    tier=1, category="keyword_overlap", label="warning", score=0.4, details="Very low"
                ),
            ],
        ),
    ))

    return rules


# =============================================================================
# Error Pattern Analyzer Tests
# =============================================================================

class TestErrorPatternAnalyzer:
    """Test error pattern analysis."""

    def test_analyze_rules(self, rules_with_consistency):
        """Test basic rule analysis."""
        analyzer = ErrorPatternAnalyzer()
        stats = analyzer.analyze_rules(rules_with_consistency)

        assert "schema_valid" in stats
        assert "deontic_alignment" in stats
        assert "keyword_overlap" in stats

    def test_category_stats(self, rules_with_consistency):
        """Test category statistics computation."""
        analyzer = ErrorPatternAnalyzer()
        stats = analyzer.analyze_rules(rules_with_consistency)

        schema_stats = stats["schema_valid"]
        assert schema_stats.total == 4  # 4 rules with consistency blocks have this check
        assert schema_stats.pass_count == 4
        assert schema_stats.fail_count == 0
        assert schema_stats.pass_rate == 1.0

        deontic_stats = stats["deontic_alignment"]
        assert deontic_stats.total == 3
        assert deontic_stats.pass_count == 1
        assert deontic_stats.warning_count == 1
        assert deontic_stats.fail_count == 1

    def test_build_error_matrix(self, rules_with_consistency):
        """Test error matrix construction."""
        analyzer = ErrorPatternAnalyzer()
        matrix = analyzer.build_error_matrix(rules_with_consistency)

        assert "schema_valid" in matrix
        assert matrix["schema_valid"]["pass"] == 4

        assert "deontic_alignment" in matrix
        assert matrix["deontic_alignment"]["pass"] == 1
        assert matrix["deontic_alignment"]["warning"] == 1
        assert matrix["deontic_alignment"]["fail"] == 1

    def test_detect_patterns(self, rules_with_consistency):
        """Test pattern detection."""
        analyzer = ErrorPatternAnalyzer()
        patterns = analyzer.detect_patterns(rules_with_consistency, min_affected=1)

        # Should detect patterns for categories with warnings/failures
        pattern_ids = [p.pattern_id for p in patterns]

        # keyword_overlap has 2 warnings
        assert any("keyword_overlap" in p for p in pattern_ids)

    def test_build_review_queue(self, rules_with_consistency):
        """Test review queue construction."""
        analyzer = ErrorPatternAnalyzer()
        queue = analyzer.build_review_queue(rules_with_consistency)

        assert len(queue) > 0

        # Inconsistent should be first (highest priority)
        assert queue[0].rule_id == "rule_inconsistent"
        assert queue[0].status == ConsistencyStatus.INCONSISTENT

        # Unverified should be in queue
        unverified_in_queue = any(q.rule_id == "rule_unverified" for q in queue)
        assert unverified_in_queue

    def test_get_summary_stats(self, rules_with_consistency):
        """Test summary statistics."""
        analyzer = ErrorPatternAnalyzer()
        summary = analyzer.get_summary_stats(rules_with_consistency)

        assert summary["total_rules"] == 5
        assert summary["verified"] == 1
        assert summary["needs_review"] == 2
        assert summary["inconsistent"] == 1
        assert summary["unverified"] == 1
        assert "timestamp" in summary


# =============================================================================
# Drift Detector Tests
# =============================================================================

class TestDriftDetector:
    """Test drift detection."""

    def test_capture_metrics(self, rules_with_consistency):
        """Test metrics capture."""
        detector = DriftDetector()
        metrics = detector.capture_metrics(rules_with_consistency)

        assert isinstance(metrics, DriftMetrics)
        assert metrics.total_rules == 5
        assert metrics.verified_count == 1
        assert metrics.needs_review_count == 2
        assert metrics.inconsistent_count == 1
        assert metrics.unverified_count == 1
        assert metrics.avg_confidence > 0

    def test_set_baseline(self, rules_with_consistency):
        """Test baseline setting."""
        detector = DriftDetector()
        baseline = detector.set_baseline(rules=rules_with_consistency)

        assert baseline is not None
        assert detector._baseline == baseline

    def test_detect_no_drift(self, rules_with_consistency):
        """Test drift detection when no drift."""
        detector = DriftDetector()
        detector.set_baseline(rules=rules_with_consistency)

        # Same rules = no drift
        report = detector.detect_drift(rules_with_consistency)

        assert isinstance(report, DriftReport)
        assert report.drift_severity == "none"
        assert not report.drift_detected

    def test_detect_drift_with_degradation(self):
        """Test drift detection with degradation."""
        detector = DriftDetector()

        # Baseline: good state
        good_rules = [
            Rule(
                rule_id="rule1",
                source=SourceRef(document_id="test"),
                consistency=ConsistencyBlock(
                    summary=ConsistencySummary(
                        status=ConsistencyStatus.VERIFIED,
                        confidence=0.95,
                    ),
                    evidence=[
                        ConsistencyEvidence(
                            tier=0, category="schema_valid", label="pass", score=1.0, details="OK"
                        ),
                    ],
                ),
            ),
            Rule(
                rule_id="rule2",
                source=SourceRef(document_id="test"),
                consistency=ConsistencyBlock(
                    summary=ConsistencySummary(
                        status=ConsistencyStatus.VERIFIED,
                        confidence=0.9,
                    ),
                    evidence=[
                        ConsistencyEvidence(
                            tier=0, category="schema_valid", label="pass", score=1.0, details="OK"
                        ),
                    ],
                ),
            ),
        ]
        detector.set_baseline(rules=good_rules)

        # Current: degraded state
        bad_rules = [
            Rule(
                rule_id="rule1",
                source=SourceRef(document_id="test"),
                consistency=ConsistencyBlock(
                    summary=ConsistencySummary(
                        status=ConsistencyStatus.INCONSISTENT,
                        confidence=0.3,
                    ),
                    evidence=[
                        ConsistencyEvidence(
                            tier=0, category="schema_valid", label="fail", score=0.0, details="Failed"
                        ),
                    ],
                ),
            ),
            Rule(
                rule_id="rule2",
                source=SourceRef(document_id="test"),
                consistency=ConsistencyBlock(
                    summary=ConsistencySummary(
                        status=ConsistencyStatus.NEEDS_REVIEW,
                        confidence=0.5,
                    ),
                    evidence=[
                        ConsistencyEvidence(
                            tier=0, category="schema_valid", label="warning", score=0.5, details="Warn"
                        ),
                    ],
                ),
            ),
        ]

        report = detector.detect_drift(bad_rules)

        assert report.drift_detected
        assert report.drift_severity in ("moderate", "major")

    def test_get_history(self, rules_with_consistency):
        """Test history tracking."""
        detector = DriftDetector()

        # Capture multiple snapshots
        detector.capture_metrics(rules_with_consistency)
        detector.capture_metrics(rules_with_consistency)
        detector.capture_metrics(rules_with_consistency)

        history = detector.get_history()
        assert len(history) == 3

    def test_get_trend(self, rules_with_consistency):
        """Test trend extraction."""
        detector = DriftDetector()

        # Capture multiple snapshots
        detector.capture_metrics(rules_with_consistency)
        detector.capture_metrics(rules_with_consistency)

        trend = detector.get_trend("avg_confidence", window=5)
        assert len(trend) == 2
        assert all(isinstance(t, tuple) for t in trend)

    def test_compare_authors(self, rules_with_consistency):
        """Test author comparison."""
        detector = DriftDetector()
        author_stats = detector.compare_authors(rules_with_consistency)

        assert "system" in author_stats
        assert "human:reviewer1" in author_stats

        assert author_stats["system"]["rule_count"] >= 2
        assert "avg_score" in author_stats["system"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestAnalyticsIntegration:
    """Test integration between analytics components."""

    def test_analyzer_with_loader(self, tmp_path):
        """Test analyzer with RuleLoader."""
        loader = RuleLoader(tmp_path)

        # Add rules to loader
        rule = Rule(
            rule_id="test_rule",
            source=SourceRef(document_id="test"),
            decision_tree=DecisionLeaf(result="test"),
            consistency=ConsistencyBlock(
                summary=ConsistencySummary(
                    status=ConsistencyStatus.VERIFIED,
                    confidence=0.9,
                ),
                evidence=[
                    ConsistencyEvidence(
                        tier=0, category="schema_valid", label="pass", score=1.0, details="OK"
                    ),
                ],
            ),
        )
        loader._rules[rule.rule_id] = rule

        analyzer = ErrorPatternAnalyzer(rule_loader=loader)
        stats = analyzer.analyze_rules()

        assert "schema_valid" in stats

    def test_drift_with_loader(self, tmp_path):
        """Test drift detector with RuleLoader."""
        loader = RuleLoader(tmp_path)

        rule = Rule(
            rule_id="test_rule",
            source=SourceRef(document_id="test"),
            decision_tree=DecisionLeaf(result="test"),
        )
        loader._rules[rule.rule_id] = rule

        detector = DriftDetector(rule_loader=loader)
        metrics = detector.capture_metrics()

        assert metrics.total_rules == 1
        assert metrics.unverified_count == 1

    def test_full_workflow(self, rules_with_consistency):
        """Test full analytics workflow."""
        # 1. Analyze patterns
        analyzer = ErrorPatternAnalyzer()
        stats = analyzer.analyze_rules(rules_with_consistency)
        patterns = analyzer.detect_patterns(rules_with_consistency)
        queue = analyzer.build_review_queue(rules_with_consistency)
        summary = analyzer.get_summary_stats(rules_with_consistency)

        # 2. Set up drift detection
        detector = DriftDetector()
        detector.set_baseline(rules=rules_with_consistency)

        # 3. Simulate some changes (just use same rules for test)
        drift_report = detector.detect_drift(rules_with_consistency)

        # Verify outputs
        assert len(stats) > 0
        assert len(queue) > 0
        assert summary["total_rules"] == 5
        assert drift_report.drift_severity == "none"
