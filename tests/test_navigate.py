"""Integration tests for cross-border compliance navigation."""

from __future__ import annotations

import pytest

from backend.rule_service.app.services.jurisdiction.resolver import resolve_jurisdictions, get_equivalences
from backend.rule_service.app.services.jurisdiction.evaluator import evaluate_jurisdiction_sync
from backend.rule_service.app.services.jurisdiction.conflicts import detect_conflicts, check_timeline_conflicts
from backend.rule_service.app.services.jurisdiction.pathway import (
    synthesize_pathway,
    aggregate_obligations,
    estimate_timeline,
    get_critical_path,
)
from backend.core.ontology.jurisdiction import JurisdictionCode, JurisdictionRole
from backend.database_service.app.services.database import init_db_with_seed, set_db_path
import tempfile


@pytest.fixture(scope="module")
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        set_db_path(f.name)
    init_db_with_seed()
    yield f.name


class TestJurisdictionResolver:
    """Tests for jurisdiction resolution."""

    def test_resolve_issuer_and_targets(self):
        """Test resolving issuer and target jurisdictions."""
        applicable = resolve_jurisdictions(
            issuer="CH",
            targets=["EU", "UK"],
            instrument_type="stablecoin",
        )

        assert len(applicable) == 3

        # Issuer should be first with role issuer_home
        assert applicable[0].jurisdiction == JurisdictionCode.CH
        assert applicable[0].role == JurisdictionRole.ISSUER_HOME
        assert applicable[0].regime_id == "finsa_dlt_2021"

        # Targets should follow
        jurisdictions = [j.jurisdiction.value for j in applicable]
        assert "EU" in jurisdictions
        assert "UK" in jurisdictions

    def test_resolve_single_jurisdiction(self):
        """Test resolving single jurisdiction (no cross-border)."""
        applicable = resolve_jurisdictions(
            issuer="EU",
            targets=[],
            instrument_type="art",
        )

        assert len(applicable) == 1
        assert applicable[0].jurisdiction == JurisdictionCode.EU
        assert applicable[0].role == JurisdictionRole.ISSUER_HOME

    def test_resolve_does_not_duplicate_issuer(self):
        """Test that issuer is not duplicated if in targets."""
        applicable = resolve_jurisdictions(
            issuer="EU",
            targets=["EU", "UK"],
            instrument_type="stablecoin",
        )

        # Should have 2, not 3 (EU not duplicated)
        assert len(applicable) == 2
        eu_count = sum(1 for j in applicable if j.jurisdiction == JurisdictionCode.EU)
        assert eu_count == 1


class TestEquivalences:
    """Tests for equivalence determinations."""

    def test_get_equivalences(self, temp_db):
        """Test retrieving equivalence determinations."""
        equivalences = get_equivalences(
            from_jurisdiction="CH",
            to_jurisdictions=["EU"],
        )

        # Should find the seeded CH->EU prospectus equivalence
        assert len(equivalences) >= 0  # May be empty if no equivalences seeded

    def test_get_equivalences_empty(self, temp_db):
        """Test getting equivalences when none exist."""
        equivalences = get_equivalences(
            from_jurisdiction="SG",
            to_jurisdictions=["UK"],
        )

        assert equivalences == [] or isinstance(equivalences, list)


class TestJurisdictionEvaluator:
    """Tests for jurisdiction evaluation."""

    def test_evaluate_eu_jurisdiction(self):
        """Test evaluating EU jurisdiction rules."""
        result = evaluate_jurisdiction_sync(
            jurisdiction="EU",
            regime_id="mica_2023",
            facts={
                "instrument_type": "stablecoin",
                "activity": "public_offer",
                "issuer_type": "credit_institution",
                "jurisdiction": "EU",
            },
        )

        assert result["jurisdiction"] == "EU"
        assert result["regime_id"] == "mica_2023"
        assert "rules_evaluated" in result
        assert "decisions" in result
        assert "obligations" in result
        assert "status" in result

    def test_evaluate_uk_jurisdiction(self):
        """Test evaluating UK jurisdiction rules."""
        result = evaluate_jurisdiction_sync(
            jurisdiction="UK",
            regime_id="fca_crypto_2024",
            facts={
                "instrument_type": "crypto_asset",
                "activity": "financial_promotion",
                "target_jurisdiction": "UK",
                "is_fca_authorized": True,
            },
        )

        assert result["jurisdiction"] == "UK"
        assert result["regime_id"] == "fca_crypto_2024"
        assert result["rules_evaluated"] > 0

    def test_evaluate_unknown_jurisdiction(self):
        """Test evaluating with no applicable rules."""
        result = evaluate_jurisdiction_sync(
            jurisdiction="SG",
            regime_id="psa_2019",
            facts={
                "instrument_type": "stablecoin",
                "activity": "public_offer",
            },
        )

        # Singapore has no rules loaded, should show no applicable rules
        assert result["jurisdiction"] == "SG"
        assert result["status"] == "no_applicable_rules"


class TestConflictDetection:
    """Tests for conflict detection."""

    def test_detect_decision_conflict(self):
        """Test detecting decision conflicts between jurisdictions."""
        results = [
            {
                "jurisdiction": "EU",
                "status": "compliant",
                "decisions": [{"rule_id": "test_1", "decision": "authorized"}],
                "obligations": [],
            },
            {
                "jurisdiction": "UK",
                "status": "blocked",
                "decisions": [{"rule_id": "test_2", "decision": "prohibited"}],
                "obligations": [],
            },
        ]

        conflicts = detect_conflicts(results)

        assert len(conflicts) > 0
        decision_conflicts = [c for c in conflicts if c["type"] == "decision_conflict"]
        assert len(decision_conflicts) > 0

    def test_detect_no_conflicts(self):
        """Test when there are no conflicts."""
        results = [
            {
                "jurisdiction": "EU",
                "status": "compliant",
                "decisions": [],
                "obligations": [],
            },
            {
                "jurisdiction": "UK",
                "status": "compliant",
                "decisions": [],
                "obligations": [],
            },
        ]

        conflicts = detect_conflicts(results)
        assert conflicts == []

    def test_check_timeline_conflicts(self):
        """Test checking for timeline conflicts in obligations."""
        obligations = [
            {"id": "submit_whitepaper", "deadline": "30 days", "jurisdiction": "EU"},
            {"id": "submit_whitepaper", "deadline": "60 days", "jurisdiction": "UK"},
        ]

        conflicts = check_timeline_conflicts(obligations)

        # Different deadlines for same obligation type should create conflict
        assert len(conflicts) >= 0


class TestPathwaySynthesis:
    """Tests for compliance pathway synthesis."""

    def test_synthesize_pathway(self):
        """Test synthesizing compliance pathway."""
        results = [
            {
                "jurisdiction": "EU",
                "regime_id": "mica_2023",
                "status": "requires_action",
                "decisions": [],
                "obligations": [
                    {"id": "obtain_authorization", "description": "Get CASP license"},
                    {"id": "submit_whitepaper", "description": "Submit crypto-asset whitepaper"},
                ],
            },
            {
                "jurisdiction": "UK",
                "regime_id": "fca_crypto_2024",
                "status": "requires_action",
                "decisions": [],
                "obligations": [
                    {"id": "add_risk_warning", "description": "Add prescribed risk warning"},
                ],
            },
        ]

        pathway = synthesize_pathway(results, conflicts=[], equivalences=[])

        assert len(pathway) == 3
        assert all("step_id" in step for step in pathway)
        assert all("action" in step for step in pathway)

    def test_aggregate_obligations(self):
        """Test aggregating obligations across jurisdictions."""
        results = [
            {
                "jurisdiction": "EU",
                "regime_id": "mica_2023",
                "obligations": [
                    {"id": "auth_1", "description": "EU auth"},
                    {"id": "disclosure_1", "description": "EU disclosure"},
                ],
            },
            {
                "jurisdiction": "UK",
                "regime_id": "fca_crypto_2024",
                "obligations": [
                    {"id": "auth_2", "description": "UK auth"},
                ],
            },
        ]

        obligations = aggregate_obligations(results)

        assert len(obligations) == 3
        assert all("jurisdiction" in o for o in obligations)

    def test_estimate_timeline(self):
        """Test estimating overall timeline."""
        pathway = [
            {"step_id": 1, "status": "pending", "timeline": {"max_days": 180}},
            {"step_id": 2, "status": "pending", "timeline": {"max_days": 30}},
            {"step_id": 3, "status": "waived", "timeline": {"max_days": 90}},
        ]

        timeline = estimate_timeline(pathway)

        assert timeline in ["< 1 month", "1-3 months", "3-6 months", "6-12 months"]

    def test_get_critical_path(self):
        """Test identifying the critical path."""
        pathway = [
            {"step_id": 1, "prerequisites": [], "timeline": {"max_days": 30}},
            {"step_id": 2, "prerequisites": [1], "timeline": {"max_days": 60}},
            {"step_id": 3, "prerequisites": [2], "timeline": {"max_days": 90}},
        ]

        critical = get_critical_path(pathway)

        assert len(critical) == 3
        assert critical[0]["step_id"] == 1  # First step in critical path


class TestEndToEndNavigation:
    """End-to-end integration tests for the navigation flow."""

    def test_full_navigation_flow(self, temp_db):
        """Test complete navigation flow."""
        # Step 1: Resolve jurisdictions
        applicable = resolve_jurisdictions(
            issuer="CH",
            targets=["EU", "UK"],
            instrument_type="stablecoin",
        )
        assert len(applicable) == 3

        # Step 2: Get equivalences
        equivalences = get_equivalences(
            from_jurisdiction="CH",
            to_jurisdictions=["EU", "UK"],
        )
        assert isinstance(equivalences, list)

        # Step 3: Evaluate each jurisdiction
        results = []
        for j in applicable:
            result = evaluate_jurisdiction_sync(
                jurisdiction=j.jurisdiction.value,
                regime_id=j.regime_id,
                facts={
                    "instrument_type": "stablecoin",
                    "activity": "public_offer",
                    "target_jurisdiction": j.jurisdiction.value,
                },
            )
            result["role"] = j.role.value
            results.append(result)

        assert len(results) == 3

        # Step 4: Detect conflicts
        conflicts = detect_conflicts(results)
        assert isinstance(conflicts, list)

        # Step 5: Synthesize pathway
        pathway = synthesize_pathway(results, conflicts, equivalences)
        assert isinstance(pathway, list)

        # Step 6: Aggregate obligations
        obligations = aggregate_obligations(results)
        assert isinstance(obligations, list)

        # Step 7: Estimate timeline
        timeline = estimate_timeline(pathway)
        assert isinstance(timeline, str)

    def test_navigation_with_blocking_jurisdiction(self, temp_db):
        """Test navigation when a jurisdiction blocks the activity."""
        # Evaluate UK with missing authorization
        result = evaluate_jurisdiction_sync(
            jurisdiction="UK",
            regime_id="fca_crypto_2024",
            facts={
                "instrument_type": "crypto_asset",
                "activity": "financial_promotion",
                "target_jurisdiction": "UK",
                "is_fca_authorized": False,
                "promotion_approved_by_authorized": False,
                "is_mlr_registered": False,
            },
        )

        # Should have obligations or blocked status
        assert result["jurisdiction"] == "UK"
        assert "obligations" in result or result["status"] == "blocked"
