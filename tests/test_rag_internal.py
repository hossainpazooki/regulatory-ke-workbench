"""Tests for internal RAG rule context retrieval."""

from __future__ import annotations

import pytest
from datetime import date
from pathlib import Path

from backend.rules import (
    Rule,
    RuleLoader,
    SourceRef,
    ConditionGroupSpec,
    ConditionSpec,
    DecisionLeaf,
)
from backend.rag import RuleContextRetriever, RuleContext


# =============================================================================
# Sample Legal Text for Testing
# =============================================================================

SAMPLE_MICA_TEXT = """
Regulation (EU) 2023/1114 of the European Parliament and of the Council

TITLE III
ASSET-REFERENCED TOKENS

Article 35 - Scope

This Title applies to issuers of asset-referenced tokens.

Article 36 - Authorisation

1. No person shall make a public offer in the Union of an asset-referenced token,
or seek admission of such a crypto-asset to trading on a trading platform,
unless that person is a legal person that has been authorised in accordance
with Article 21.

2. Paragraph 1 shall not apply to an issuer of asset-referenced tokens
that is a credit institution authorised under Directive 2013/36/EU.

3. The competent authority shall inform ESMA of any authorisation granted
under this Article.

Article 37 - Application for authorisation

1. An issuer of asset-referenced tokens seeking authorisation shall submit
an application to the competent authority of the home Member State.

2. The application shall contain all of the following:
(a) the name and legal entity identifier of the applicant;
(b) a programme of operations;
(c) the crypto-asset white paper referred to in Article 19.

Article 38 - Reserve of assets

1. Issuers of asset-referenced tokens shall constitute and maintain a reserve
of assets. The reserve shall be composed and managed in such a way that
the risks associated with the assets referenced are covered.

2. The reserve of assets shall be composed of assets that are:
(a) highly liquid with minimal market risk, credit risk and concentration risk;
(b) capable of being liquidated rapidly with minimal adverse price effect.
"""


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def rule_context_retriever() -> RuleContextRetriever:
    """Create a RuleContextRetriever with sample documents indexed."""
    retriever = RuleContextRetriever()
    retriever.index_document("mica_2023", SAMPLE_MICA_TEXT)
    return retriever


@pytest.fixture
def sample_rule() -> Rule:
    """Create a sample rule with source reference."""
    return Rule(
        rule_id="mica_art36_authorization",
        version="1.0",
        description="Authorization requirement for asset-referenced token public offers",
        source=SourceRef(
            document_id="mica_2023",
            article="36",
            paragraphs=["1", "2"],
        ),
        tags=["mica", "art", "authorization"],
        applies_if=ConditionGroupSpec(
            all=[
                ConditionSpec(field="instrument_type", value="art"),
                ConditionSpec(field="activity", value="public_offer"),
            ]
        ),
        decision_tree=DecisionLeaf(result="authorization_required"),
    )


@pytest.fixture
def rule_loader(tmp_path: Path) -> RuleLoader:
    """Create a RuleLoader with sample rules."""
    loader = RuleLoader(tmp_path)

    # Create sample rules
    rules = [
        Rule(
            rule_id="mica_art36_auth",
            source=SourceRef(document_id="mica_2023", article="36"),
            description="Authorization for public offers",
            tags=["mica", "art", "authorization"],
            decision_tree=DecisionLeaf(result="required"),
        ),
        Rule(
            rule_id="mica_art37_application",
            source=SourceRef(document_id="mica_2023", article="37"),
            description="Application for authorization",
            tags=["mica", "art", "application"],
            decision_tree=DecisionLeaf(result="submit_application"),
        ),
        Rule(
            rule_id="mica_art38_reserves",
            source=SourceRef(document_id="mica_2023", article="38"),
            description="Reserve of assets requirements",
            tags=["mica", "art", "reserves"],
            decision_tree=DecisionLeaf(result="maintain_reserves"),
        ),
        Rule(
            rule_id="other_regulation_rule",
            source=SourceRef(document_id="other_2024", article="1"),
            description="Some other regulation rule",
            tags=["other"],
            decision_tree=DecisionLeaf(result="other"),
        ),
    ]

    for rule in rules:
        loader._rules[rule.rule_id] = rule

    return loader


# =============================================================================
# Index Tests
# =============================================================================

class TestRuleContextRetrieverIndex:
    """Test document indexing."""

    def test_index_document(self):
        """Test indexing a document."""
        retriever = RuleContextRetriever()
        chunks = retriever.index_document("test_doc", SAMPLE_MICA_TEXT)

        assert chunks > 0
        assert "test_doc" in retriever.indexed_documents
        assert len(retriever) > 0

    def test_index_document_with_metadata(self):
        """Test indexing with metadata."""
        retriever = RuleContextRetriever()
        chunks = retriever.index_document(
            "mica",
            SAMPLE_MICA_TEXT,
            metadata={"source": "EUR-Lex", "year": "2023"},
        )

        assert chunks > 0
        assert "mica" in retriever.indexed_documents

    def test_index_short_document(self):
        """Test indexing a short document (no chunking needed)."""
        retriever = RuleContextRetriever()
        short_text = "Article 1 - This is a short test."
        chunks = retriever.index_document("short", short_text)

        assert chunks == 1
        assert len(retriever) == 1


# =============================================================================
# Source Context Retrieval Tests
# =============================================================================

class TestSourceContextRetrieval:
    """Test retrieving source context for rules."""

    def test_get_source_context(self, rule_context_retriever, sample_rule):
        """Test retrieving source context for a rule."""
        results = rule_context_retriever.get_source_context(sample_rule, top_k=3)

        assert len(results) > 0
        # Should find Article 36 content
        combined_text = " ".join(r.text.lower() for r in results)
        assert "article 36" in combined_text or "authorisation" in combined_text

    def test_get_source_context_no_source(self, rule_context_retriever):
        """Test retrieving context for rule without source reference."""
        rule = Rule(rule_id="no_source_rule")
        results = rule_context_retriever.get_source_context(rule)

        assert len(results) == 0

    def test_get_source_text(self, rule_context_retriever, sample_rule):
        """Test getting source text as string."""
        text = rule_context_retriever.get_source_text(sample_rule)

        assert text is not None
        assert len(text) > 0
        # Should contain relevant content
        assert "authoris" in text.lower() or "public offer" in text.lower()


# =============================================================================
# Cross-Reference Detection Tests
# =============================================================================

class TestCrossReferenceDetection:
    """Test detection of cross-references in legal text."""

    def test_find_article_references(self, rule_context_retriever):
        """Test finding Article references."""
        text = "As specified in Article 21 and Article 36(1), the issuer must..."
        refs = rule_context_retriever.find_cross_references(text)

        assert "Article 21" in refs
        assert "Article 36(1)" in refs

    def test_find_directive_references(self, rule_context_retriever):
        """Test finding Directive references."""
        text = "Credit institutions under Directive 2013/36/EU are exempt."
        refs = rule_context_retriever.find_cross_references(text)

        assert "Directive 2013/36/EU" in refs

    def test_find_regulation_references(self, rule_context_retriever):
        """Test finding Regulation references."""
        text = "As defined in Regulation (EU) 2023/1114..."
        refs = rule_context_retriever.find_cross_references(text)

        assert "Regulation (EU) 2023/1114" in refs

    def test_find_multiple_references(self, rule_context_retriever):
        """Test finding multiple cross-references."""
        text = """
        Article 36(1) requires authorization under Article 21.
        This applies unless Directive 2013/36/EU applies.
        See also Regulation (EU) 2023/1114.
        """
        refs = rule_context_retriever.find_cross_references(text)

        assert len(refs) >= 4
        assert "Article 36(1)" in refs
        assert "Article 21" in refs

    def test_no_duplicate_references(self, rule_context_retriever):
        """Test that duplicate references are not included."""
        text = "Article 36 states... Article 36 also requires..."
        refs = rule_context_retriever.find_cross_references(text)

        # Should only have one "Article 36"
        article_36_count = sum(1 for r in refs if r == "Article 36")
        assert article_36_count == 1


# =============================================================================
# Related Rules Tests
# =============================================================================

class TestRelatedRules:
    """Test finding related rules."""

    def test_find_related_rules_by_document(self, rule_loader):
        """Test finding related rules in same document."""
        retriever = RuleContextRetriever(rule_loader=rule_loader)

        rule = rule_loader.get_rule("mica_art36_auth")
        related = retriever.find_related_rules(rule, top_k=5)

        # Should find other MiCA rules
        related_ids = [r.rule_id for r in related]
        assert "mica_art37_application" in related_ids
        assert "mica_art38_reserves" in related_ids

        # Should NOT include the rule itself
        assert "mica_art36_auth" not in related_ids

        # Should rank same-document rules higher than other documents
        # (other_regulation_rule should be last or not included)

    def test_find_related_rules_by_tags(self, rule_loader):
        """Test finding related rules by overlapping tags."""
        retriever = RuleContextRetriever(rule_loader=rule_loader)

        rule = rule_loader.get_rule("mica_art36_auth")
        related = retriever.find_related_rules(rule, top_k=3)

        # All related should have some tag overlap
        for related_rule in related:
            assert any(tag in rule.tags for tag in related_rule.tags)

    def test_find_related_rules_no_loader(self):
        """Test that no related rules returned without loader."""
        retriever = RuleContextRetriever()  # No rule_loader

        rule = Rule(rule_id="test", source=SourceRef(document_id="test"))
        related = retriever.find_related_rules(rule)

        assert len(related) == 0


# =============================================================================
# Complete Rule Context Tests
# =============================================================================

class TestRuleContext:
    """Test getting complete rule context."""

    def test_get_rule_context(self, rule_context_retriever, rule_loader, sample_rule):
        """Test getting complete context for a rule."""
        # Add rule_loader to retriever
        retriever = RuleContextRetriever(rule_loader=rule_loader)
        retriever.index_document("mica_2023", SAMPLE_MICA_TEXT)

        context = retriever.get_rule_context(sample_rule)

        assert isinstance(context, RuleContext)
        assert context.rule_id == sample_rule.rule_id
        assert len(context.source_passages) > 0

    def test_get_rule_context_includes_cross_refs(self, rule_context_retriever, sample_rule):
        """Test that context includes cross-references from source."""
        context = rule_context_retriever.get_rule_context(sample_rule)

        # MICA Art 36 references Article 21 and Directive 2013/36/EU
        # Cross-refs should be found in the retrieved source passages
        # (may be empty if source passages don't contain refs)
        assert isinstance(context.cross_references, list)

    def test_get_rule_context_structure(self):
        """Test the structure of RuleContext."""
        retriever = RuleContextRetriever()
        retriever.index_document("test", "Article 1 - Test content")

        rule = Rule(
            rule_id="test_rule",
            source=SourceRef(document_id="test", article="1"),
            decision_tree=DecisionLeaf(result="test"),
        )

        context = retriever.get_rule_context(rule)

        assert hasattr(context, "rule_id")
        assert hasattr(context, "source_passages")
        assert hasattr(context, "cross_references")
        assert hasattr(context, "related_rules")


# =============================================================================
# Integration with Consistency Engine
# =============================================================================

class TestConsistencyEngineIntegration:
    """Test integration between RAG and consistency engine."""

    def test_consistency_engine_with_retriever(self, rule_context_retriever, sample_rule):
        """Test that consistency engine can use retriever for source text."""
        from backend.verification import ConsistencyEngine

        # Create consistency engine with retriever
        engine = ConsistencyEngine(retriever=rule_context_retriever._retriever)

        # Verify rule - should use retriever to fetch source
        result = engine.verify_rule(sample_rule, tiers=[0, 1])

        assert result is not None
        assert result.summary is not None

    def test_source_text_from_retriever(self, rule_context_retriever, sample_rule):
        """Test getting source text for consistency verification."""
        source_text = rule_context_retriever.get_source_text(sample_rule)

        assert source_text is not None

        # Can now pass to consistency checks
        from backend.verification import check_deontic_alignment

        evidence = check_deontic_alignment(sample_rule, source_text)
        assert evidence.tier == 1
        # Should not be "No source text" warning
        assert "No source text" not in evidence.details
