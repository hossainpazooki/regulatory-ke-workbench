"""Tests for rule context retrieval with legal corpus."""

import pytest

from backend.rag.frontend_helpers import (
    get_rule_context,
    RuleContextPayload,
    _get_rule_loader,
    _get_context_retriever,
)


class TestRuleContextWithLegalCorpus:
    """Tests for rule context retrieval using legal corpus."""

    def test_get_rule_context_mica_rule(self):
        """Test getting context for a MiCA rule."""
        # This rule should exist and reference mica_2023
        ctx = get_rule_context("mica_art36_public_offer_authorization")

        if ctx is None:
            pytest.skip("MiCA rule not loaded")

        assert ctx.rule_id == "mica_art36_public_offer_authorization"
        assert ctx.document_id == "mica_2023"

        # Should have document metadata from legal corpus
        assert ctx.document_title is not None
        assert "MiCA" in ctx.document_title or "Markets in Crypto" in ctx.document_title

        assert ctx.citation is not None
        assert "2023" in ctx.citation or "EU" in ctx.citation

        assert ctx.source_url is not None
        assert ctx.source_url.startswith("http")

    def test_get_rule_context_includes_article_text(self):
        """Test that context includes article text from legal corpus."""
        ctx = get_rule_context("mica_art36_public_offer_authorization")

        if ctx is None:
            pytest.skip("MiCA rule not loaded")

        # Primary span should contain some legal text
        assert ctx.primary_span
        assert len(ctx.primary_span) > 50

        # Should contain Article 36 content (about authorization)
        span_lower = ctx.primary_span.lower()
        # At least one of these should appear in the article text
        assert any(word in span_lower for word in [
            "authoris", "author", "offer", "public", "asset",
        ])

    def test_get_rule_context_missing_rule(self):
        """Test getting context for non-existent rule."""
        ctx = get_rule_context("nonexistent_rule_xyz")
        assert ctx is None

    def test_context_retriever_indexes_legal_corpus(self):
        """Test that context retriever indexes legal documents."""
        retriever = _get_context_retriever()

        # Should have indexed legal documents
        indexed = retriever.indexed_documents

        assert "mica_2023" in indexed
        assert "dlt_pilot_2022" in indexed
        assert "genius_act_2025" in indexed

    def test_context_retriever_has_documents(self):
        """Test that retriever has indexed chunks."""
        retriever = _get_context_retriever()

        # Should have indexed multiple chunks
        assert len(retriever) > 0


class TestRuleContextPayloadStructure:
    """Tests for RuleContextPayload structure."""

    def test_payload_has_all_fields(self):
        """Test that payload includes all expected fields."""
        ctx = get_rule_context("mica_art36_public_offer_authorization")

        if ctx is None:
            pytest.skip("MiCA rule not loaded")

        # Required fields
        assert hasattr(ctx, "rule_id")
        assert hasattr(ctx, "document_id")
        assert hasattr(ctx, "article")
        assert hasattr(ctx, "section")
        assert hasattr(ctx, "pages")
        assert hasattr(ctx, "primary_span")
        assert hasattr(ctx, "before")
        assert hasattr(ctx, "after")

        # Legal corpus metadata fields
        assert hasattr(ctx, "document_title")
        assert hasattr(ctx, "citation")
        assert hasattr(ctx, "source_url")


class TestRuleContextForDifferentDocuments:
    """Tests for rule context across different legal documents."""

    def test_context_for_each_document_type(self):
        """Test that rules from different documents get appropriate context."""
        loader = _get_rule_loader()
        rules = loader.get_all_rules()

        # Group rules by document_id
        rules_by_doc = {}
        for rule in rules:
            if rule.source:
                doc_id = rule.source.document_id
                if doc_id not in rules_by_doc:
                    rules_by_doc[doc_id] = []
                rules_by_doc[doc_id].append(rule)

        # Test at least one rule from mica_2023
        if "mica_2023" in rules_by_doc:
            rule = rules_by_doc["mica_2023"][0]
            ctx = get_rule_context(rule.rule_id)

            assert ctx is not None
            assert ctx.document_id == "mica_2023"
            assert ctx.document_title is not None
