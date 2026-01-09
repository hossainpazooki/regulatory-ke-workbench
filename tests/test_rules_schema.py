"""Tests for rule schema with consistency blocks."""

from __future__ import annotations

import pytest
from pathlib import Path
from datetime import date

from backend.rule_service.app.services.loader import RuleLoader, Rule
from backend.rule_service.app.services.schema import (
    ConsistencyBlock,
    ConsistencySummary,
    ConsistencyEvidence,
    ConsistencyStatus,
    SourceRef,
    ConditionGroup,
    ConditionSpec,
    DecisionBranch,
    DecisionLeaf,
    Rule as SchemaRule,
)


class TestConsistencyModels:
    """Test consistency-related Pydantic models."""

    def test_consistency_evidence_creation(self):
        """Test creating a ConsistencyEvidence instance."""
        evidence = ConsistencyEvidence(
            tier=0,
            category="schema_valid",
            label="pass",
            score=1.0,
            details="All required fields present",
        )
        assert evidence.tier == 0
        assert evidence.category == "schema_valid"
        assert evidence.label == "pass"
        assert evidence.score == 1.0
        assert evidence.timestamp  # Should have default

    def test_consistency_summary_defaults(self):
        """Test ConsistencySummary default values."""
        summary = ConsistencySummary()
        assert summary.status == ConsistencyStatus.UNVERIFIED
        assert summary.confidence == 0.0
        assert summary.last_verified is None

    def test_consistency_summary_with_values(self):
        """Test ConsistencySummary with explicit values."""
        summary = ConsistencySummary(
            status=ConsistencyStatus.VERIFIED,
            confidence=0.95,
            last_verified="2024-12-10T14:30:00Z",
            verified_by="system",
            notes="Verified via Tier 0-1 checks",
        )
        assert summary.status == ConsistencyStatus.VERIFIED
        assert summary.confidence == 0.95
        assert summary.verified_by == "system"

    def test_consistency_block_creation(self):
        """Test creating a complete ConsistencyBlock."""
        evidence = [
            ConsistencyEvidence(
                tier=0,
                category="schema_valid",
                label="pass",
                score=1.0,
                details="Schema validation passed",
            ),
            ConsistencyEvidence(
                tier=1,
                category="deontic_alignment",
                label="pass",
                score=0.9,
                details="Deontic verbs align with rule modality",
                source_span="shall make a public offer",
                rule_element="applies_if.all[0]",
            ),
        ]
        summary = ConsistencySummary(
            status=ConsistencyStatus.VERIFIED,
            confidence=0.95,
        )
        block = ConsistencyBlock(summary=summary, evidence=evidence)

        assert block.summary.status == ConsistencyStatus.VERIFIED
        assert len(block.evidence) == 2
        assert block.evidence[0].tier == 0
        assert block.evidence[1].tier == 1


class TestRuleWithConsistency:
    """Test Rule model with consistency blocks."""

    def test_rule_without_consistency(self, rule_loader: RuleLoader):
        """Test loading a rule without consistency block."""
        rule = rule_loader.get_rule("mica_art36_public_offer_authorization")
        assert rule is not None
        assert rule.consistency is None

    def test_rule_with_consistency_programmatic(self):
        """Test creating a rule with consistency block programmatically."""
        consistency = ConsistencyBlock(
            summary=ConsistencySummary(
                status=ConsistencyStatus.VERIFIED,
                confidence=0.92,
                last_verified="2024-12-10T14:30:00Z",
                verified_by="system",
            ),
            evidence=[
                ConsistencyEvidence(
                    tier=0,
                    category="required_fields",
                    label="pass",
                    score=1.0,
                    details="All required fields present",
                ),
            ],
        )

        # Create rule with consistency
        from backend.rule_service.app.services.loader import (
            Rule,
            SourceRef,
            ConditionGroupSpec,
            ConditionSpec,
            DecisionLeaf,
            ObligationSpec,
        )

        rule = Rule(
            rule_id="test_rule_with_consistency",
            version="1.0",
            description="Test rule with consistency block",
            effective_from=date(2024, 1, 1),
            tags=["test"],
            applies_if=ConditionGroupSpec(
                all=[ConditionSpec(field="test_field", operator="==", value=True)]
            ),
            decision_tree=DecisionLeaf(result="test_result"),
            source=SourceRef(document_id="test_doc", article="1"),
            consistency=consistency,
        )

        assert rule.consistency is not None
        assert rule.consistency.summary.status == ConsistencyStatus.VERIFIED
        assert len(rule.consistency.evidence) == 1


class TestSourceRefExtensions:
    """Test extended SourceRef fields."""

    def test_source_ref_with_paragraphs(self):
        """Test SourceRef with paragraph references."""
        from backend.rule_service.app.services.loader import SourceRef

        source = SourceRef(
            document_id="mica_2023",
            article="36(1)",
            paragraphs=["1", "2", "3"],
            pages=[65, 66],
            url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1114",
        )
        assert source.document_id == "mica_2023"
        assert len(source.paragraphs) == 3
        assert source.url is not None


class TestDecisionEngineWithConsistency:
    """Test that decision engine passes through consistency metadata."""

    def test_decision_result_has_rule_metadata(self, decision_engine):
        """Test that DecisionResult includes rule_metadata."""
        from backend.core.ontology import Scenario

        scenario = Scenario(
            instrument_type="art",
            activity="public_offer",
            jurisdiction="EU",
            authorized=False,
            is_credit_institution=False,
        )
        result = decision_engine.evaluate(scenario, "mica_art36_public_offer_authorization")

        assert result.rule_metadata is not None
        assert result.rule_metadata.rule_id == "mica_art36_public_offer_authorization"
        assert result.rule_metadata.version == "1.0"
        assert result.rule_metadata.source is not None
        assert "mica" in result.rule_metadata.tags

    def test_rule_metadata_consistency_none_when_not_present(self, decision_engine):
        """Test that consistency is None when rule has no consistency block."""
        from backend.core.ontology import Scenario

        scenario = Scenario(
            instrument_type="art",
            activity="public_offer",
            jurisdiction="EU",
            authorized=True,
        )
        result = decision_engine.evaluate(scenario, "mica_art36_public_offer_authorization")

        assert result.rule_metadata is not None
        # Rule doesn't have consistency block yet
        assert result.rule_metadata.consistency is None


class TestRuleSaveLoad:
    """Test saving and loading rules with consistency."""

    def test_save_rule_with_consistency(self, tmp_path: Path):
        """Test saving a rule with consistency block to YAML."""
        from backend.rule_service.app.services.loader import (
            Rule,
            RuleLoader,
            SourceRef,
            DecisionLeaf,
        )

        # Create rule with consistency
        consistency = ConsistencyBlock(
            summary=ConsistencySummary(
                status=ConsistencyStatus.NEEDS_REVIEW,
                confidence=0.75,
                verified_by="system",
            ),
            evidence=[
                ConsistencyEvidence(
                    tier=0,
                    category="schema_valid",
                    label="pass",
                    score=1.0,
                    details="Schema valid",
                ),
                ConsistencyEvidence(
                    tier=1,
                    category="deontic_alignment",
                    label="warning",
                    score=0.6,
                    details="Deontic verb mismatch detected",
                    source_span="may offer",
                    rule_element="decision_tree.result",
                ),
            ],
        )

        rule = Rule(
            rule_id="test_save_consistency",
            version="1.0",
            description="Test rule for save/load",
            decision_tree=DecisionLeaf(result="test_result"),
            source=SourceRef(document_id="test_doc", article="1"),
            consistency=consistency,
        )

        # Save
        loader = RuleLoader(tmp_path)
        path = loader.save_rule(rule)
        assert path.exists()

        # Load back
        loader2 = RuleLoader(tmp_path)
        loaded_rules = loader2.load_file(path)
        assert len(loaded_rules) == 1

        loaded = loaded_rules[0]
        assert loaded.rule_id == "test_save_consistency"
        assert loaded.consistency is not None
        assert loaded.consistency.summary.status == ConsistencyStatus.NEEDS_REVIEW
        assert len(loaded.consistency.evidence) == 2
        assert loaded.consistency.evidence[1].label == "warning"
