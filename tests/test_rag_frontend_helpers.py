"""Tests for RAG frontend helpers."""

from __future__ import annotations

import pytest

from backend.rag.frontend_helpers import (
    RuleContextPayload,
    RelatedProvision,
    SearchResult,
    get_rule_context,
    get_related_provisions,
    search_corpus,
    _parse_article_reference,
    _normalize_article,
)


# =============================================================================
# Article Pattern Tests
# =============================================================================


class TestArticlePatternParsing:
    """Test article reference pattern matching."""

    def test_parse_art_dot_number(self):
        """Parse 'Art. 36(1)' format."""
        result = _parse_article_reference("Art. 36(1)")
        assert result == ("36", "1")

    def test_parse_art_number_no_paragraph(self):
        """Parse 'Art. 36' format without paragraph."""
        result = _parse_article_reference("Art. 36")
        assert result == ("36", None)

    def test_parse_article_full(self):
        """Parse 'Article 45(2)' format."""
        result = _parse_article_reference("Article 45(2)")
        assert result == ("45", "2")

    def test_parse_article_no_paragraph(self):
        """Parse 'Article 45' format."""
        result = _parse_article_reference("Article 45")
        assert result == ("45", None)

    def test_parse_bare_number(self):
        """Parse '36(1)' bare format."""
        result = _parse_article_reference("36(1)")
        assert result == ("36", "1")

    def test_parse_bare_number_no_paragraph(self):
        """Parse '36' bare format."""
        result = _parse_article_reference("36")
        assert result == ("36", None)

    def test_parse_not_article_reference(self):
        """Non-article queries return None."""
        result = _parse_article_reference("reserve assets")
        assert result is None

    def test_parse_mixed_query_not_article(self):
        """Mixed queries that don't start with article pattern."""
        result = _parse_article_reference("what does article 36 say about")
        assert result is None  # Doesn't match our patterns (too much text)


class TestArticleNormalization:
    """Test article normalization."""

    def test_normalize_simple_number(self):
        """Normalize simple article number."""
        assert _normalize_article("36") == "36"

    def test_normalize_with_paragraph(self):
        """Normalize article with paragraph."""
        assert _normalize_article("36(1)") == "36"

    def test_normalize_with_prefix(self):
        """Normalize article with 'Art.' prefix."""
        assert _normalize_article("Art. 36") == "36"

    def test_normalize_none(self):
        """None returns None."""
        assert _normalize_article(None) is None


# =============================================================================
# Rule Context Tests
# =============================================================================


class TestGetRuleContext:
    """Test get_rule_context function."""

    def test_known_rule_returns_payload(self):
        """Known MiCA rule returns RuleContextPayload."""
        ctx = get_rule_context("mica_art36_public_offer_authorization")

        if ctx is None:
            pytest.skip("Rule not available in test environment")

        assert isinstance(ctx, RuleContextPayload)
        assert ctx.rule_id == "mica_art36_public_offer_authorization"
        assert ctx.document_id  # Should have a document_id
        assert ctx.primary_span  # Should have primary span text

    def test_known_rule_has_matching_source(self):
        """Rule context has matching document_id and article."""
        ctx = get_rule_context("mica_art36_public_offer_authorization")

        if ctx is None:
            pytest.skip("Rule not available in test environment")

        # Should have MiCA document ID
        assert "mica" in ctx.document_id.lower()
        # Should have article reference
        assert ctx.article is not None
        assert "36" in ctx.article

    def test_unknown_rule_returns_none(self):
        """Unknown rule returns None."""
        ctx = get_rule_context("nonexistent_rule_xyz_12345")
        assert ctx is None

    def test_context_payload_structure(self):
        """Verify RuleContextPayload has expected fields."""
        ctx = get_rule_context("mica_art36_public_offer_authorization")

        if ctx is None:
            pytest.skip("Rule not available in test environment")

        # Check all expected fields are present
        assert hasattr(ctx, "rule_id")
        assert hasattr(ctx, "document_id")
        assert hasattr(ctx, "article")
        assert hasattr(ctx, "section")
        assert hasattr(ctx, "pages")
        assert hasattr(ctx, "primary_span")
        assert hasattr(ctx, "before")
        assert hasattr(ctx, "after")

        # before and after should be lists
        assert isinstance(ctx.before, list)
        assert isinstance(ctx.after, list)


# =============================================================================
# Related Provisions Tests
# =============================================================================


class TestGetRelatedProvisions:
    """Test get_related_provisions function."""

    def test_returns_list(self):
        """Related provisions returns a list."""
        provisions = get_related_provisions("mica_art36_public_offer_authorization")
        assert isinstance(provisions, list)

    def test_high_threshold_can_return_empty(self):
        """Very high threshold (0.99) can return empty list."""
        provisions = get_related_provisions(
            "mica_art36_public_offer_authorization",
            threshold=0.99,
        )
        # Should not error, may be empty
        assert isinstance(provisions, list)

    def test_low_threshold_may_return_results(self):
        """Lower threshold may return results."""
        provisions = get_related_provisions(
            "mica_art36_public_offer_authorization",
            threshold=0.3,
            limit=5,
        )
        assert isinstance(provisions, list)

    def test_limit_respected(self):
        """Limit parameter is respected."""
        provisions = get_related_provisions(
            "mica_art36_public_offer_authorization",
            threshold=0.1,
            limit=3,
        )
        assert len(provisions) <= 3

    def test_unknown_rule_returns_empty(self):
        """Unknown rule returns empty list."""
        provisions = get_related_provisions("nonexistent_rule_xyz_12345")
        assert provisions == []

    def test_provision_structure(self):
        """Verify RelatedProvision has expected fields."""
        provisions = get_related_provisions(
            "mica_art36_public_offer_authorization",
            threshold=0.3,
            limit=5,
        )

        for provision in provisions:
            assert isinstance(provision, RelatedProvision)
            assert hasattr(provision, "document_id")
            assert hasattr(provision, "article")
            assert hasattr(provision, "snippet")
            assert hasattr(provision, "score")
            assert hasattr(provision, "rule_id")

            # Score should be a float between 0 and 1
            assert isinstance(provision.score, float)
            assert 0.0 <= provision.score <= 1.0


# =============================================================================
# Corpus Search Tests
# =============================================================================


class TestSearchCorpus:
    """Test search_corpus function."""

    def test_article_reference_returns_article_mode(self):
        """Article reference query returns mode='article'."""
        result = search_corpus("Art. 36(1)")
        assert isinstance(result, SearchResult)
        assert result.mode == "article"

    def test_article_36_mode(self):
        """Searching for Article 36 returns article mode."""
        result = search_corpus("Article 36")
        assert result.mode == "article"

    def test_natural_language_returns_semantic_mode(self):
        """Natural language query returns mode='semantic'."""
        result = search_corpus("reserve assets")
        assert isinstance(result, SearchResult)
        assert result.mode == "semantic"

    def test_empty_query_returns_semantic_empty(self):
        """Empty query returns empty semantic result."""
        result = search_corpus("")
        assert result.mode == "semantic"
        assert result.semantic_hits == []

    def test_article_mode_has_article_hits(self):
        """Article mode populates article_hits list."""
        result = search_corpus("Art. 36")

        # article_hits should be a list
        assert isinstance(result.article_hits, list)
        # semantic_hits should be empty for article mode
        assert result.semantic_hits == []

    def test_semantic_mode_has_semantic_hits(self):
        """Semantic mode populates semantic_hits list."""
        result = search_corpus("authorization requirements")

        # semantic_hits should be a list
        assert isinstance(result.semantic_hits, list)
        # article_hits should be empty for semantic mode
        assert result.article_hits == []

    def test_article_hit_structure(self):
        """Verify ArticleHit has expected fields."""
        result = search_corpus("Art. 36")

        for hit in result.article_hits:
            assert hasattr(hit, "rule_id")
            assert hasattr(hit, "document_id")
            assert hasattr(hit, "article")
            assert hasattr(hit, "primary_span")
            assert hasattr(hit, "description")

    def test_semantic_hit_structure(self):
        """Verify SemanticHit has expected fields."""
        result = search_corpus("reserve assets")

        for hit in result.semantic_hits:
            assert hasattr(hit, "document_id")
            assert hasattr(hit, "article")
            assert hasattr(hit, "snippet")
            assert hasattr(hit, "score")
            assert hasattr(hit, "rule_id")

    def test_max_hits_respected(self):
        """max_hits parameter is respected."""
        result = search_corpus("authorization", max_hits=3)

        if result.mode == "article":
            assert len(result.article_hits) <= 3
        else:
            assert len(result.semantic_hits) <= 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the frontend helpers."""

    def test_context_and_related_consistency(self):
        """Context and related provisions work together."""
        rule_id = "mica_art36_public_offer_authorization"

        ctx = get_rule_context(rule_id)
        if ctx is None:
            pytest.skip("Rule not available")

        related = get_related_provisions(rule_id, threshold=0.3)

        # If we have related provisions, they should have document_id
        for provision in related:
            assert provision.document_id is not None

    def test_search_then_get_context(self):
        """Search result can be used to get context."""
        result = search_corpus("Art. 36")

        for hit in result.article_hits:
            # Each hit's rule_id should work with get_rule_context
            ctx = get_rule_context(hit.rule_id)
            assert ctx is not None
            assert ctx.rule_id == hit.rule_id
