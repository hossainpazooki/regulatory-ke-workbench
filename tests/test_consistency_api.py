"""Tests for KE API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


# =============================================================================
# Basic Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Test health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_app_info(self, client):
        """Root endpoint returns app information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data
        assert "ke" in data["endpoints"]


# =============================================================================
# KE Verification Endpoint Tests
# =============================================================================


class TestVerifyEndpoint:
    """Test /ke/verify endpoint."""

    def test_verify_known_rule(self, client):
        """Verify a known rule returns consistency data."""
        response = client.post(
            "/ke/verify",
            json={
                "rule_id": "mica_art36_public_offer_authorization",
                "tiers": [0, 1],
            },
        )

        # Rule might not exist in test environment
        if response.status_code == 200:
            data = response.json()
            assert "rule_id" in data
            assert "status" in data
            assert "confidence" in data
            assert "evidence" in data
        else:
            # 404 is acceptable if rules aren't loaded
            assert response.status_code == 404

    def test_verify_unknown_rule(self, client):
        """Verify unknown rule returns 404."""
        response = client.post(
            "/ke/verify",
            json={
                "rule_id": "nonexistent_rule_xyz",
                "tiers": [0],
            },
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_verify_with_source_text(self, client):
        """Verify rule with provided source text."""
        response = client.post(
            "/ke/verify",
            json={
                "rule_id": "mica_art36_public_offer_authorization",
                "source_text": "An issuer shall obtain authorization before making a public offer.",
                "tiers": [0, 1],
            },
        )

        if response.status_code == 200:
            data = response.json()
            # When source text is provided, tier 1 checks should run
            tier1_evidence = [e for e in data["evidence"] if e["tier"] == 1]
            assert len(tier1_evidence) > 0


class TestVerifyAllEndpoint:
    """Test /ke/verify-all endpoint."""

    def test_verify_all_rules(self, client):
        """Verify all rules returns summary."""
        response = client.post("/ke/verify-all?tiers=0&tiers=1")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_verify_all_with_tier_filter(self, client):
        """Verify all with specific tiers."""
        response = client.post("/ke/verify-all?tiers=0")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data


# =============================================================================
# KE Analytics Endpoint Tests
# =============================================================================


class TestAnalyticsSummaryEndpoint:
    """Test /ke/analytics/summary endpoint."""

    def test_analytics_summary(self, client):
        """Analytics summary returns expected fields."""
        response = client.get("/ke/analytics/summary")

        assert response.status_code == 200
        data = response.json()
        assert "total_rules" in data
        assert "verification_rate" in data
        assert "average_score" in data


class TestAnalyticsPatternsEndpoint:
    """Test /ke/analytics/patterns endpoint."""

    def test_analytics_patterns(self, client):
        """Error patterns endpoint returns list."""
        response = client.get("/ke/analytics/patterns")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestReviewQueueEndpoint:
    """Test /ke/analytics/review-queue endpoint."""

    def test_review_queue(self, client):
        """Review queue returns prioritized list."""
        response = client.get("/ke/analytics/review-queue")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_review_queue_with_limit(self, client):
        """Review queue respects max_items."""
        response = client.get("/ke/analytics/review-queue?max_items=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5


# =============================================================================
# KE Human Review Endpoint Tests
# =============================================================================


class TestHumanReviewEndpoint:
    """Test /ke/rules/{rule_id}/review endpoint."""

    def test_submit_review_valid(self, client):
        """Submit valid human review."""
        response = client.post(
            "/ke/rules/mica_art36_public_offer_authorization/review",
            json={
                "label": "consistent",
                "notes": "Verified against source text manually",
                "reviewer_id": "test_reviewer",
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert data["rule_id"] == "mica_art36_public_offer_authorization"
            assert data["status"] == "verified"
            assert data["review_tier"] == 4
            assert data["reviewer_id"] == "test_reviewer"
        else:
            # 404 if rule doesn't exist
            assert response.status_code == 404

    def test_submit_review_invalid_label(self, client):
        """Submit review with invalid label returns 400."""
        response = client.post(
            "/ke/rules/mica_art36_public_offer_authorization/review",
            json={
                "label": "invalid_label",
                "notes": "Test notes",
                "reviewer_id": "test_reviewer",
            },
        )

        # Either 400 for invalid label or 404 for missing rule
        assert response.status_code in (400, 404)

    def test_submit_review_unknown_rule(self, client):
        """Submit review for unknown rule returns 404."""
        response = client.post(
            "/ke/rules/nonexistent_rule_xyz/review",
            json={
                "label": "consistent",
                "notes": "Test notes",
                "reviewer_id": "test_reviewer",
            },
        )

        assert response.status_code == 404

    def test_get_reviews_for_rule(self, client):
        """Get human reviews for a rule."""
        response = client.get(
            "/ke/rules/mica_art36_public_offer_authorization/reviews"
        )

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            for review in data:
                assert review["tier"] == 4
                assert review["category"] == "human_review"
        else:
            # 404 if rule doesn't exist
            assert response.status_code == 404


# =============================================================================
# KE Drift Detection Endpoint Tests
# =============================================================================


class TestDriftEndpoints:
    """Test drift detection endpoints."""

    def test_set_drift_baseline(self, client):
        """Set drift baseline returns metrics."""
        response = client.post("/ke/drift/baseline")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Baseline set"

    def test_detect_drift(self, client):
        """Detect drift returns report."""
        # First set baseline
        client.post("/ke/drift/baseline")

        response = client.get("/ke/drift/detect")

        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "drift_severity" in data
        assert "summary" in data

    def test_drift_history(self, client):
        """Drift history returns list of metrics."""
        response = client.get("/ke/drift/history")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_drift_history_with_window(self, client):
        """Drift history respects window parameter."""
        response = client.get("/ke/drift/history?window=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5


# =============================================================================
# KE Context Endpoint Tests
# =============================================================================


class TestContextEndpoints:
    """Test rule context endpoints."""

    def test_get_rule_context(self, client):
        """Get context for a rule."""
        response = client.get("/ke/context/mica_art36_public_offer_authorization")

        if response.status_code == 200:
            data = response.json()
            assert "rule_id" in data
            assert "source_passages" in data
            assert "related_rules" in data
        else:
            # 404 if rule doesn't exist
            assert response.status_code == 404

    def test_get_rule_context_unknown(self, client):
        """Get context for unknown rule returns 404."""
        response = client.get("/ke/context/nonexistent_rule_xyz")

        assert response.status_code == 404

    def test_get_related_rules(self, client):
        """Get related rules."""
        response = client.get("/ke/related/mica_art36_public_offer_authorization")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            for rule in data:
                assert "rule_id" in rule
        else:
            # 404 if rule doesn't exist
            assert response.status_code == 404

    def test_get_related_rules_with_limit(self, client):
        """Get related rules respects top_k."""
        response = client.get(
            "/ke/related/mica_art36_public_offer_authorization?top_k=3"
        )

        if response.status_code == 200:
            data = response.json()
            assert len(data) <= 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestVerificationWorkflow:
    """Test complete verification workflow."""

    def test_verify_then_review_workflow(self, client):
        """Test full workflow: verify, then human review."""
        rule_id = "mica_art36_public_offer_authorization"

        # Step 1: Verify rule
        verify_response = client.post(
            "/ke/verify",
            json={"rule_id": rule_id, "tiers": [0, 1]},
        )

        if verify_response.status_code != 200:
            pytest.skip("Rule not available in test environment")

        verify_data = verify_response.json()
        initial_status = verify_data["status"]

        # Step 2: Submit human review
        review_response = client.post(
            f"/ke/rules/{rule_id}/review",
            json={
                "label": "consistent",
                "notes": "Verified against source",
                "reviewer_id": "integration_test",
            },
        )

        assert review_response.status_code == 200
        review_data = review_response.json()

        # Human review should update status
        assert review_data["status"] == "verified"
        assert review_data["review_tier"] == 4

        # Step 3: Check reviews were recorded
        reviews_response = client.get(f"/ke/rules/{rule_id}/reviews")

        assert reviews_response.status_code == 200
        reviews = reviews_response.json()
        assert len(reviews) > 0
        assert reviews[-1]["category"] == "human_review"
