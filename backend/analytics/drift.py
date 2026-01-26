"""Drift detection for rule quality monitoring.

Tracks consistency scores over time to detect:
- Degradation in rule quality after bulk updates
- Systematic issues introduced by specific authors
- Areas needing re-verification after source document updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.rules import Rule, RuleLoader, ConsistencyStatus


@dataclass
class DriftMetrics:
    """Metrics for a single time point."""

    timestamp: str
    total_rules: int
    verified_count: int
    needs_review_count: int
    inconsistent_count: int
    unverified_count: int
    avg_confidence: float
    category_scores: dict[str, float] = field(default_factory=dict)
    rules_changed: list[str] = field(default_factory=list)


@dataclass
class DriftReport:
    """Report on detected drift."""

    report_id: str
    generated_at: str
    baseline: DriftMetrics
    current: DriftMetrics
    drift_detected: bool
    drift_severity: str  # "none", "minor", "moderate", "major"
    degraded_categories: list[str]
    improved_categories: list[str]
    rules_degraded: list[str]
    rules_improved: list[str]
    summary: str


class DriftDetector:
    """Detects drift in rule quality over time.

    Compares current consistency metrics against a baseline to identify
    degradation or improvement in rule quality.
    """

    def __init__(
        self,
        rule_loader: RuleLoader | None = None,
        threshold_degradation: float = 0.1,
        threshold_improvement: float = 0.1,
    ):
        """Initialize the drift detector.

        Args:
            rule_loader: Optional rule loader to analyze rules from.
            threshold_degradation: Score drop threshold to flag degradation.
            threshold_improvement: Score increase threshold to flag improvement.
        """
        self._rule_loader = rule_loader
        self._threshold_degradation = threshold_degradation
        self._threshold_improvement = threshold_improvement
        self._history: list[DriftMetrics] = []
        self._baseline: DriftMetrics | None = None

    def capture_metrics(
        self,
        rules: list[Rule] | None = None,
        label: str | None = None,
    ) -> DriftMetrics:
        """Capture current metrics snapshot.

        Args:
            rules: Rules to analyze. If None, uses rule_loader.
            label: Optional label for this snapshot.

        Returns:
            DriftMetrics for current state.
        """
        if rules is None:
            if self._rule_loader is None:
                return self._empty_metrics()
            rules = self._rule_loader.get_all_rules()

        total = len(rules)
        verified = 0
        needs_review = 0
        inconsistent = 0
        unverified = 0
        confidence_sum = 0.0
        confidence_count = 0
        category_scores: dict[str, list[float]] = {}
        rules_with_consistency: list[str] = []

        for rule in rules:
            if not rule.consistency:
                unverified += 1
                continue

            rules_with_consistency.append(rule.rule_id)
            status = rule.consistency.summary.status

            if status == ConsistencyStatus.VERIFIED:
                verified += 1
            elif status == ConsistencyStatus.NEEDS_REVIEW:
                needs_review += 1
            elif status == ConsistencyStatus.INCONSISTENT:
                inconsistent += 1
            else:
                unverified += 1

            confidence_sum += rule.consistency.summary.confidence
            confidence_count += 1

            # Track category scores
            for ev in rule.consistency.evidence:
                if ev.category not in category_scores:
                    category_scores[ev.category] = []
                category_scores[ev.category].append(ev.score)

        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0

        # Average category scores
        avg_category_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
            if scores
        }

        metrics = DriftMetrics(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            total_rules=total,
            verified_count=verified,
            needs_review_count=needs_review,
            inconsistent_count=inconsistent,
            unverified_count=unverified,
            avg_confidence=round(avg_confidence, 4),
            category_scores=avg_category_scores,
            rules_changed=rules_with_consistency,
        )

        self._history.append(metrics)
        return metrics

    def set_baseline(
        self,
        metrics: DriftMetrics | None = None,
        rules: list[Rule] | None = None,
    ) -> DriftMetrics:
        """Set baseline metrics for comparison.

        Args:
            metrics: Metrics to use as baseline. If None, captures current.
            rules: Rules to capture baseline from (if metrics is None).

        Returns:
            The baseline metrics.
        """
        if metrics is None:
            metrics = self.capture_metrics(rules)

        self._baseline = metrics
        return metrics

    def detect_drift(
        self,
        rules: list[Rule] | None = None,
    ) -> DriftReport:
        """Detect drift from baseline.

        Args:
            rules: Current rules to compare against baseline.

        Returns:
            DriftReport with analysis.
        """
        if self._baseline is None:
            # Auto-set baseline if not set
            self.set_baseline()

        current = self.capture_metrics(rules)
        baseline = self._baseline

        # Calculate changes
        confidence_delta = current.avg_confidence - baseline.avg_confidence
        verified_delta = current.verified_count - baseline.verified_count
        inconsistent_delta = current.inconsistent_count - baseline.inconsistent_count

        # Find degraded and improved categories
        degraded_categories = []
        improved_categories = []

        for cat, score in current.category_scores.items():
            baseline_score = baseline.category_scores.get(cat, score)
            delta = score - baseline_score

            if delta < -self._threshold_degradation:
                degraded_categories.append(cat)
            elif delta > self._threshold_improvement:
                improved_categories.append(cat)

        # Find rules that degraded or improved
        rules_degraded = []
        rules_improved = []

        # Compare rule sets if we have that info
        current_rules = set(current.rules_changed)
        baseline_rules = set(baseline.rules_changed)

        new_rules = current_rules - baseline_rules
        removed_rules = baseline_rules - current_rules

        # Determine severity
        if (
            inconsistent_delta > 2 or
            confidence_delta < -0.15 or
            len(degraded_categories) > 3
        ):
            drift_severity = "major"
        elif (
            inconsistent_delta > 0 or
            confidence_delta < -0.1 or
            len(degraded_categories) > 1
        ):
            drift_severity = "moderate"
        elif (
            confidence_delta < -0.05 or
            len(degraded_categories) > 0
        ):
            drift_severity = "minor"
        else:
            drift_severity = "none"

        drift_detected = drift_severity != "none"

        # Generate summary
        summary = self._generate_summary(
            baseline=baseline,
            current=current,
            degraded_categories=degraded_categories,
            improved_categories=improved_categories,
            drift_severity=drift_severity,
        )

        return DriftReport(
            report_id=f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            baseline=baseline,
            current=current,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            degraded_categories=degraded_categories,
            improved_categories=improved_categories,
            rules_degraded=list(removed_rules)[:10],  # Rules removed might indicate issues
            rules_improved=list(new_rules)[:10],  # New rules with consistency
            summary=summary,
        )

    def get_history(self) -> list[DriftMetrics]:
        """Get history of captured metrics."""
        return self._history.copy()

    def get_trend(
        self,
        metric: str = "avg_confidence",
        window: int = 5,
    ) -> list[tuple[str, float]]:
        """Get trend for a metric over recent history.

        Args:
            metric: Which metric to track.
            window: How many snapshots to include.

        Returns:
            List of (timestamp, value) tuples.
        """
        history = self._history[-window:] if window > 0 else self._history

        trend = []
        for metrics in history:
            if metric == "avg_confidence":
                value = metrics.avg_confidence
            elif metric == "verified_count":
                value = metrics.verified_count
            elif metric == "inconsistent_count":
                value = metrics.inconsistent_count
            elif metric == "total_rules":
                value = metrics.total_rules
            else:
                # Try category score
                value = metrics.category_scores.get(metric, 0.0)

            trend.append((metrics.timestamp, value))

        return trend

    def compare_authors(
        self,
        rules: list[Rule] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare consistency metrics by author/verified_by.

        Args:
            rules: Rules to analyze.

        Returns:
            Dictionary mapping author to their metrics.
        """
        if rules is None:
            if self._rule_loader is None:
                return {}
            rules = self._rule_loader.get_all_rules()

        author_stats: dict[str, dict[str, Any]] = {}

        for rule in rules:
            if not rule.consistency:
                continue

            author = rule.consistency.summary.verified_by or "unknown"

            if author not in author_stats:
                author_stats[author] = {
                    "rule_count": 0,
                    "total_score": 0.0,
                    "verified": 0,
                    "needs_review": 0,
                    "inconsistent": 0,
                }

            stats = author_stats[author]
            stats["rule_count"] += 1
            stats["total_score"] += rule.consistency.summary.confidence

            status = rule.consistency.summary.status
            if status == ConsistencyStatus.VERIFIED:
                stats["verified"] += 1
            elif status == ConsistencyStatus.NEEDS_REVIEW:
                stats["needs_review"] += 1
            elif status == ConsistencyStatus.INCONSISTENT:
                stats["inconsistent"] += 1

        # Calculate averages
        for author, stats in author_stats.items():
            count = stats["rule_count"]
            if count > 0:
                stats["avg_score"] = round(stats["total_score"] / count, 4)
                del stats["total_score"]

        return author_stats

    def _empty_metrics(self) -> DriftMetrics:
        """Create empty metrics object."""
        return DriftMetrics(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            total_rules=0,
            verified_count=0,
            needs_review_count=0,
            inconsistent_count=0,
            unverified_count=0,
            avg_confidence=0.0,
        )

    def _generate_summary(
        self,
        baseline: DriftMetrics,
        current: DriftMetrics,
        degraded_categories: list[str],
        improved_categories: list[str],
        drift_severity: str,
    ) -> str:
        """Generate human-readable summary of drift analysis."""
        parts = []

        # Overall drift assessment
        if drift_severity == "none":
            parts.append("No significant drift detected.")
        elif drift_severity == "minor":
            parts.append("Minor drift detected - some quality metrics have decreased slightly.")
        elif drift_severity == "moderate":
            parts.append("Moderate drift detected - quality metrics show noticeable degradation.")
        else:
            parts.append("MAJOR drift detected - significant quality degradation requires immediate attention.")

        # Confidence change
        conf_delta = current.avg_confidence - baseline.avg_confidence
        if abs(conf_delta) > 0.01:
            direction = "increased" if conf_delta > 0 else "decreased"
            parts.append(
                f"Average confidence {direction} from {baseline.avg_confidence:.2%} to {current.avg_confidence:.2%}."
            )

        # Verification status changes
        verified_delta = current.verified_count - baseline.verified_count
        if verified_delta != 0:
            direction = "increased" if verified_delta > 0 else "decreased"
            parts.append(f"Verified rules {direction} by {abs(verified_delta)}.")

        inconsistent_delta = current.inconsistent_count - baseline.inconsistent_count
        if inconsistent_delta > 0:
            parts.append(f"WARNING: {inconsistent_delta} more rules now marked inconsistent.")

        # Category changes
        if degraded_categories:
            parts.append(f"Degraded categories: {', '.join(degraded_categories)}.")

        if improved_categories:
            parts.append(f"Improved categories: {', '.join(improved_categories)}.")

        return " ".join(parts)
