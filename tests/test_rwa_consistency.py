"""Tests for RWA rule consistency checks.

This module tests that the consistency engine (Tier 0-1 checks)
works correctly with RWA rules.
"""

import pytest
from pathlib import Path

from backend.rules import RuleLoader
from backend.verification import ConsistencyEngine


@pytest.fixture
def rule_loader():
    """Create a rule loader with RWA rules loaded."""
    loader = RuleLoader()
    rules_dir = Path(__file__).parent.parent / "backend" / "rule_service" / "data"
    loader.load_directory(rules_dir)
    return loader


@pytest.fixture
def consistency_engine():
    """Create a consistency engine."""
    return ConsistencyEngine()


class TestRWATier0SchemaChecks:
    """Tests for Tier 0 schema checks on RWA rules."""

    def test_tokenization_rule_passes_schema_checks(
        self, rule_loader, consistency_engine
    ):
        """Test that tokenization rule passes Tier 0 schema checks."""
        rule = rule_loader.get_rule("rwa_tokenization_authorization")
        result = consistency_engine.verify_rule(rule, tiers=[0])

        # Should have some evidence
        assert len(result.evidence) > 0

        # Check for schema validation passes
        schema_evidence = [e for e in result.evidence if e.tier == 0]
        assert len(schema_evidence) > 0

        # At least some checks should pass
        passes = [e for e in schema_evidence if e.label == "pass"]
        assert len(passes) > 0

    def test_disclosure_rule_passes_schema_checks(
        self, rule_loader, consistency_engine
    ):
        """Test that disclosure rule passes Tier 0 schema checks."""
        rule = rule_loader.get_rule("rwa_disclosure_requirements")
        result = consistency_engine.verify_rule(rule, tiers=[0])

        schema_evidence = [e for e in result.evidence if e.tier == 0]
        passes = [e for e in schema_evidence if e.label == "pass"]
        assert len(passes) > 0

    def test_custody_rule_passes_schema_checks(
        self, rule_loader, consistency_engine
    ):
        """Test that custody rule passes Tier 0 schema checks."""
        rule = rule_loader.get_rule("rwa_custody_requirements")
        result = consistency_engine.verify_rule(rule, tiers=[0])

        schema_evidence = [e for e in result.evidence if e.tier == 0]
        passes = [e for e in schema_evidence if e.label == "pass"]
        assert len(passes) > 0

    def test_rwa_rules_have_required_fields(self, rule_loader, consistency_engine):
        """Test that RWA rules have required fields."""
        rwa_rules = [
            r for r in rule_loader.get_all_rules() if r.rule_id.startswith("rwa_")
        ]

        for rule in rwa_rules:
            result = consistency_engine.verify_rule(rule, tiers=[0])

            # Look for required_fields check
            required_fields_evidence = [
                e
                for e in result.evidence
                if e.tier == 0 and "required" in e.category.lower()
            ]

            # If there's a required fields check, it should pass
            for evidence in required_fields_evidence:
                # Either pass or warning is acceptable (warning might be for optional fields)
                assert evidence.label in ["pass", "warning"], (
                    f"Rule {rule.rule_id} failed required fields check: {evidence.details}"
                )

    def test_rwa_rules_have_valid_source(self, rule_loader, consistency_engine):
        """Test that RWA rules have valid source references."""
        rwa_rules = [
            r for r in rule_loader.get_all_rules() if r.rule_id.startswith("rwa_")
        ]

        for rule in rwa_rules:
            assert rule.source is not None, f"Rule {rule.rule_id} missing source"
            assert rule.source.document_id == "rwa_eu_2025"
            assert rule.source.article is not None


class TestRWATier1LexicalChecks:
    """Tests for Tier 1 lexical checks on RWA rules."""

    def test_tokenization_rule_lexical_checks(
        self, rule_loader, consistency_engine
    ):
        """Test lexical checks on tokenization rule."""
        rule = rule_loader.get_rule("rwa_tokenization_authorization")
        result = consistency_engine.verify_rule(rule, tiers=[1])

        # Should have Tier 1 evidence
        tier1_evidence = [e for e in result.evidence if e.tier == 1]
        assert len(tier1_evidence) > 0

    def test_rwa_rules_have_tags(self, rule_loader):
        """Test that RWA rules have appropriate tags."""
        rwa_rules = [
            r for r in rule_loader.get_all_rules() if r.rule_id.startswith("rwa_")
        ]

        for rule in rwa_rules:
            assert rule.tags is not None
            assert len(rule.tags) > 0
            assert "rwa" in rule.tags, f"Rule {rule.rule_id} missing 'rwa' tag"


class TestRWAVerificationSummary:
    """Tests for verification summary of RWA rules."""

    def test_verification_produces_summary(self, rule_loader, consistency_engine):
        """Test that verification produces a summary."""
        rule = rule_loader.get_rule("rwa_tokenization_authorization")
        result = consistency_engine.verify_rule(rule)

        assert result.summary is not None
        assert result.summary.status in [
            "verified",
            "needs_review",
            "inconsistent",
            "unverified",
        ]
        assert 0.0 <= result.summary.confidence <= 1.0

    def test_all_rwa_rules_verify(self, rule_loader, consistency_engine):
        """Test that all RWA rules can be verified without errors."""
        rwa_rules = [
            r for r in rule_loader.get_all_rules() if r.rule_id.startswith("rwa_")
        ]

        for rule in rwa_rules:
            # Should not raise any exceptions
            result = consistency_engine.verify_rule(rule)
            assert result is not None
            assert result.summary is not None
            assert result.evidence is not None


class TestRWAConsistencyWithSource:
    """Tests for consistency verification with source text."""

    def test_verify_with_source_text(self, rule_loader, consistency_engine):
        """Test verification when source text is provided."""
        rule = rule_loader.get_rule("rwa_tokenization_authorization")

        # Provide some source text for verification
        source_text = """
        Article 5(1) requires authorization for tokenizing real-world assets.
        Investment firms authorized under MiFID II are exempt from this requirement.
        """

        result = consistency_engine.verify_rule(rule, source_text=source_text)

        # Should have some evidence
        assert len(result.evidence) > 0

    def test_verify_all_rwa_rules_with_tiers(self, rule_loader, consistency_engine):
        """Test verification of all RWA rules at different tiers."""
        rwa_rules = [
            r for r in rule_loader.get_all_rules() if r.rule_id.startswith("rwa_")
        ]

        for rule in rwa_rules:
            # Tier 0 only
            result0 = consistency_engine.verify_rule(rule, tiers=[0])
            tier0_evidence = [e for e in result0.evidence if e.tier == 0]
            assert len(tier0_evidence) > 0

            # Tier 0 and 1
            result01 = consistency_engine.verify_rule(rule, tiers=[0, 1])
            all_evidence = result01.evidence
            assert len(all_evidence) >= len(tier0_evidence)
