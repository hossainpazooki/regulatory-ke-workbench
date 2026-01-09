"""Frontend helpers for RAG integration in KE Streamlit UI.

This module wraps the internal RAG layer with helpers for:
1. Rule source & context retrieval
2. Related provisions with structural filtering and similarity thresholds
3. Dual-mode corpus search (article lookup vs semantic search)

These helpers do NOT use external LLMs - only local retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.rule_service.app.services import RuleLoader, Rule
from .rule_context import RuleContextRetriever
from .retriever import Retriever


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RuleContextPayload:
    """Context payload for a rule, suitable for UI display."""

    rule_id: str
    document_id: str
    article: str | None
    section: str | None
    pages: list[int] | None
    primary_span: str  # main text span backing the rule
    before: list[str]  # a few preceding paragraphs
    after: list[str]  # a few following paragraphs
    # Legal corpus metadata
    document_title: str | None = None
    citation: str | None = None
    source_url: str | None = None


@dataclass
class RelatedProvision:
    """A provision related to a rule."""

    document_id: str
    article: str | None
    snippet: str
    score: float
    rule_id: str | None  # if we can map hit → rule


@dataclass
class ArticleHit:
    """A rule matching an article reference."""

    rule_id: str
    document_id: str
    article: str | None
    primary_span: str
    description: str | None


@dataclass
class SemanticHit:
    """A semantic search hit."""

    document_id: str
    article: str | None
    snippet: str
    score: float
    rule_id: str | None  # if mappable to a rule
    source_type: str | None = None  # "legal_text" or None for rule-derived
    document_title: str | None = None  # title of source document
    has_rule_coverage: bool = True  # False if legal text with no mapped rule


@dataclass
class SearchResult:
    """Result from corpus search."""

    mode: str  # "article" or "semantic"
    article_hits: list[ArticleHit] = field(default_factory=list)
    semantic_hits: list[SemanticHit] = field(default_factory=list)


# =============================================================================
# Shared State (lazy initialization)
# =============================================================================


_rule_loader: RuleLoader | None = None
_context_retriever: RuleContextRetriever | None = None


def _get_rule_loader() -> RuleLoader:
    """Get or create the rule loader."""
    global _rule_loader
    if _rule_loader is None:
        from backend.core.config import get_settings
        settings = get_settings()
        _rule_loader = RuleLoader(settings.rules_dir)
        _rule_loader.load_directory()
    return _rule_loader


def _get_context_retriever() -> RuleContextRetriever:
    """Get or create the context retriever."""
    global _context_retriever
    if _context_retriever is None:
        loader = _get_rule_loader()
        _context_retriever = RuleContextRetriever(rule_loader=loader)

        # Index legal corpus from data/legal/
        from .corpus_loader import load_all_legal_documents

        for doc in load_all_legal_documents():
            try:
                _context_retriever.index_document(
                    document_id=doc.document_id,
                    text=doc.text,
                    metadata={
                        "document_title": doc.title,
                        "citation": doc.citation,
                        "jurisdiction": doc.jurisdiction,
                        "source_url": doc.source_url,
                        "source_type": "legal_text",
                    },
                )
            except Exception:
                pass  # Ignore indexing errors

        # Also try to load any loose txt files in data/
        data_dir = Path(__file__).parent.parent.parent / "data"
        if data_dir.exists():
            for txt_file in data_dir.glob("*.txt"):
                try:
                    _context_retriever.index_document_file(txt_file)
                except Exception:
                    pass  # Ignore indexing errors

    return _context_retriever


# =============================================================================
# Article Pattern Matching
# =============================================================================


# Patterns for article references
ARTICLE_PATTERNS = [
    # "Art. 45(2)", "Art 45(2)", "Art.45(2)"
    re.compile(r"Art\.?\s*(\d+)(?:\((\d+)\))?", re.IGNORECASE),
    # "Article 45(2)", "Article 45"
    re.compile(r"Article\s+(\d+)(?:\((\d+)\))?", re.IGNORECASE),
    # Just "45(2)" or "45" at the start
    re.compile(r"^(\d+)(?:\((\d+)\))?$"),
]


def _parse_article_reference(query: str) -> tuple[str, str | None] | None:
    """Parse article reference from query.

    Args:
        query: Query string like "Art. 36(1)" or "Article 45"

    Returns:
        Tuple of (article_number, paragraph) or None if not an article reference.
    """
    query = query.strip()

    for pattern in ARTICLE_PATTERNS:
        match = pattern.match(query)
        if match:
            article = match.group(1)
            paragraph = match.group(2) if match.lastindex >= 2 else None
            return (article, paragraph)

    return None


def _normalize_article(article: str | None) -> str | None:
    """Normalize article reference for comparison.

    Extracts just the base article number, e.g.:
    - "36(1)" -> "36"
    - "Art. 36" -> "36"
    - "36" -> "36"
    """
    if not article:
        return None

    # Extract just the number
    match = re.search(r"(\d+)", article)
    if match:
        return match.group(1)
    return article


# =============================================================================
# Helper Functions
# =============================================================================


def get_rule_context(rule_id: str) -> RuleContextPayload | None:
    """Get context payload for a rule.

    Args:
        rule_id: The rule ID to get context for.

    Returns:
        RuleContextPayload or None if rule not found.
    """
    from .corpus_loader import load_legal_document, LegalCorpusError

    loader = _get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        return None

    retriever = _get_context_retriever()
    context = retriever.get_rule_context(rule)

    # Extract document metadata from rule source
    document_id = rule.source.document_id if rule.source else ""
    article = rule.source.article if rule.source else None
    section = rule.source.section if rule.source else None
    pages = rule.source.pages if rule.source else None

    # Try to get legal document metadata
    document_title = None
    citation = None
    source_url = None

    if document_id:
        try:
            legal_doc = load_legal_document(document_id)
            document_title = legal_doc.title
            citation = legal_doc.citation
            source_url = legal_doc.source_url

            # Try to find specific article text from legal corpus
            if article:
                article_text = legal_doc.find_article_text(article)
                if article_text:
                    # Use legal corpus text as primary span
                    context.source_passages = []  # Clear retrieval results
        except LegalCorpusError:
            pass  # Document not in corpus, use retrieval results

    # Build primary span from source passages or legal corpus
    primary_span = ""
    before_paragraphs: list[str] = []
    after_paragraphs: list[str] = []

    # Try to get article text directly from legal corpus
    if document_id and article:
        try:
            legal_doc = load_legal_document(document_id)
            article_text = legal_doc.find_article_text(article)
            if article_text:
                primary_span = article_text
        except LegalCorpusError:
            pass

    # Fall back to retrieval results
    if not primary_span and context.source_passages:
        # First passage is the primary span
        primary_span = context.source_passages[0].text

        # Additional passages can be before/after context
        # (In practice, we don't have true ordering, so we split them)
        for i, passage in enumerate(context.source_passages[1:], start=1):
            if i <= 2:
                before_paragraphs.append(passage.text)
            else:
                after_paragraphs.append(passage.text)

    # Fall back to rule description if no passages found
    if not primary_span:
        primary_span = rule.description or f"No source text found for {rule_id}"

    return RuleContextPayload(
        rule_id=rule_id,
        document_id=document_id,
        article=article,
        section=section,
        pages=pages,
        primary_span=primary_span,
        before=before_paragraphs,
        after=after_paragraphs,
        document_title=document_title,
        citation=citation,
        source_url=source_url,
    )


def get_related_provisions(
    rule_id: str,
    *,
    threshold: float = 0.7,
    limit: int = 10,
) -> list[RelatedProvision]:
    """Get provisions related to a rule with filtering.

    Applies structural filter (same document_id) and similarity threshold.

    Args:
        rule_id: The rule to find related provisions for.
        threshold: Minimum similarity score (0.0-1.0). Hits below this are dropped.
        limit: Maximum number of provisions to return.

    Returns:
        List of RelatedProvision, empty if none above threshold.
    """
    loader = _get_rule_loader()
    rule = loader.get_rule(rule_id)

    if rule is None:
        return []

    retriever = _get_context_retriever()
    all_rules = loader.get_all_rules()

    # Build a map from (document_id, article) to rule_ids
    source_to_rules: dict[tuple[str, str | None], list[str]] = {}
    for r in all_rules:
        if r.source:
            key = (r.source.document_id, _normalize_article(r.source.article))
            if key not in source_to_rules:
                source_to_rules[key] = []
            source_to_rules[key].append(r.rule_id)

    # Get related rules using existing logic
    related_rules = retriever.find_related_rules(rule, top_k=limit * 2)

    # Get source document_id for structural filtering
    source_document_id = rule.source.document_id if rule.source else None

    provisions: list[RelatedProvision] = []

    for related in related_rules:
        if related.rule_id == rule_id:
            continue

        # Structural filter: prefer same document
        related_doc_id = related.source.document_id if related.source else None

        # Calculate a pseudo-score based on relationship strength
        score = 0.0

        # Same document bonus
        if source_document_id and related_doc_id == source_document_id:
            score += 0.5

        # Same article bonus
        if (
            rule.source
            and related.source
            and rule.source.article
            and related.source.article
        ):
            if _normalize_article(rule.source.article) == _normalize_article(
                related.source.article
            ):
                score += 0.3

        # Tag overlap bonus
        if rule.tags and related.tags:
            overlap = len(set(rule.tags) & set(related.tags))
            score += min(overlap * 0.1, 0.2)

        # Apply threshold
        if score < threshold:
            continue

        # Build snippet from description or rule_id
        snippet = related.description or f"Rule: {related.rule_id}"
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        provisions.append(
            RelatedProvision(
                document_id=related_doc_id or "",
                article=related.source.article if related.source else None,
                snippet=snippet,
                score=round(score, 2),
                rule_id=related.rule_id,
            )
        )

    # Sort by score descending
    provisions.sort(key=lambda p: p.score, reverse=True)

    return provisions[:limit]


def search_corpus(query: str, *, max_hits: int = 15) -> SearchResult:
    """Search the corpus with dual-mode support.

    If query matches an article pattern (e.g., "Art. 36(1)"), performs
    article lookup. Otherwise, performs semantic search.

    Args:
        query: Search query (article reference or natural language).
        max_hits: Maximum number of hits to return.

    Returns:
        SearchResult with mode indicator and appropriate hits.
    """
    query = query.strip()

    if not query:
        return SearchResult(mode="semantic", semantic_hits=[])

    # Check if query is an article reference
    article_ref = _parse_article_reference(query)

    if article_ref:
        return _search_by_article(article_ref[0], article_ref[1], max_hits)
    else:
        return _search_semantic(query, max_hits)


def _search_by_article(
    article: str,
    paragraph: str | None,
    max_hits: int,
) -> SearchResult:
    """Search for rules by article reference.

    Args:
        article: Article number (e.g., "36").
        paragraph: Optional paragraph number (e.g., "1").
        max_hits: Maximum number of hits.

    Returns:
        SearchResult with mode="article" and article hits.
    """
    loader = _get_rule_loader()
    all_rules = loader.get_all_rules()

    hits: list[ArticleHit] = []

    for rule in all_rules:
        if not rule.source or not rule.source.article:
            continue

        # Normalize and compare article numbers
        rule_article = _normalize_article(rule.source.article)

        if rule_article != article:
            continue

        # If paragraph specified, check for paragraph match
        if paragraph:
            # Check if rule article contains the paragraph
            if f"({paragraph})" not in rule.source.article:
                continue

        # Get context for primary span
        ctx = get_rule_context(rule.rule_id)
        primary_span = ctx.primary_span if ctx else ""

        hits.append(
            ArticleHit(
                rule_id=rule.rule_id,
                document_id=rule.source.document_id,
                article=rule.source.article,
                primary_span=primary_span[:300] if primary_span else "",
                description=rule.description,
            )
        )

        if len(hits) >= max_hits:
            break

    return SearchResult(mode="article", article_hits=hits)


def _search_semantic(query: str, max_hits: int) -> SearchResult:
    """Perform semantic search on the corpus.

    Args:
        query: Natural language query.
        max_hits: Maximum number of hits.

    Returns:
        SearchResult with mode="semantic" and semantic hits.
    """
    retriever = _get_context_retriever()
    loader = _get_rule_loader()
    all_rules = loader.get_all_rules()

    # Build a map from (document_id, normalized_article) to rule_ids
    source_to_rules: dict[tuple[str, str | None], list[str]] = {}
    for r in all_rules:
        if r.source:
            key = (r.source.document_id, _normalize_article(r.source.article))
            if key not in source_to_rules:
                source_to_rules[key] = []
            source_to_rules[key].append(r.rule_id)

    # Search using the retriever
    results = retriever._retriever.search(query, top_k=max_hits, method="bm25")

    hits: list[SemanticHit] = []

    for result in results:
        document_id = result.document_id

        # Try to extract article from metadata or text
        article = result.metadata.get("article")
        if not article:
            # Try to extract article from the text itself
            article_match = re.search(r"Article\s+(\d+)", result.text)
            if article_match:
                article = article_match.group(1)

        # Try to map to a rule
        rule_id = None
        has_rule_coverage = True
        if document_id:
            key = (document_id, _normalize_article(article))
            matching_rules = source_to_rules.get(key, [])
            if matching_rules:
                rule_id = matching_rules[0]  # Pick first matching rule
            else:
                # Legal text hit with no corresponding rule
                has_rule_coverage = False

        # Get source type and document title from metadata
        source_type = result.metadata.get("source_type")
        document_title = result.metadata.get("document_title")

        # Truncate snippet
        snippet = result.text
        if len(snippet) > 250:
            snippet = snippet[:247] + "..."

        hits.append(
            SemanticHit(
                document_id=document_id,
                article=article,
                snippet=snippet,
                score=round(result.score, 3),
                rule_id=rule_id,
                source_type=source_type,
                document_title=document_title,
                has_rule_coverage=has_rule_coverage,
            )
        )

    return SearchResult(mode="semantic", semantic_hits=hits)


# =============================================================================
# Convenience exports
# =============================================================================


__all__ = [
    "RuleContextPayload",
    "RelatedProvision",
    "ArticleHit",
    "SemanticHit",
    "SearchResult",
    "get_rule_context",
    "get_related_provisions",
    "search_corpus",
]
