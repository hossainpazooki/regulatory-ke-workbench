"""RAG module - Retrieval-Augmented Generation for factual Q&A."""

from .retriever import Retriever, RetrievalResult
from .bm25 import BM25Index
from .chunker import chunk_text, chunk_by_section, Chunk
from .rule_context import RuleContextRetriever, RuleContext, ProvisionContext
from .corpus_loader import (
    LegalDocument,
    LegalCorpusError,
    load_legal_document,
    load_all_legal_documents,
    get_available_document_ids,
    chunk_legal_document,
    index_legal_corpus,
)
from .frontend_helpers import (
    RuleContextPayload,
    RelatedProvision,
    ArticleHit,
    SemanticHit,
    SearchResult,
    get_rule_context,
    get_related_provisions,
    search_corpus,
)

__all__ = [
    "Retriever",
    "RetrievalResult",
    "BM25Index",
    "chunk_text",
    "chunk_by_section",
    "Chunk",
    "RuleContextRetriever",
    "RuleContext",
    "ProvisionContext",
    # Legal corpus
    "LegalDocument",
    "LegalCorpusError",
    "load_legal_document",
    "load_all_legal_documents",
    "get_available_document_ids",
    "chunk_legal_document",
    "index_legal_corpus",
    # Frontend helpers
    "RuleContextPayload",
    "RelatedProvision",
    "ArticleHit",
    "SemanticHit",
    "SearchResult",
    "get_rule_context",
    "get_related_provisions",
    "search_corpus",
]
