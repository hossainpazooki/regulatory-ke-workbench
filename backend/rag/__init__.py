"""RAG domain - factual Q&A with retrieval augmented generation."""

from .router import router
from .schemas import AskRequest, AskResponse, SourceCitation
from .service import (
    # Chunking
    Chunk,
    chunk_text,
    chunk_by_section,
    # BM25
    BM25Document,
    BM25Index,
    # Retrieval
    RetrievalResult,
    Retriever,
    # Generation
    AnswerGenerator,
)

# Legal corpus loader
from .corpus_loader import (
    LegalDocument,
    LegalCorpusError,
    load_legal_document,
    load_all_legal_documents,
    get_available_document_ids,
    chunk_legal_document,
    index_legal_corpus,
)

# Rule context retriever
from .rule_context import (
    RuleContextRetriever,
    RuleContext,
    ProvisionContext,
)

# Frontend helpers
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
    # Router
    "router",
    # Schemas
    "AskRequest",
    "AskResponse",
    "SourceCitation",
    # Chunking
    "Chunk",
    "chunk_text",
    "chunk_by_section",
    # BM25
    "BM25Document",
    "BM25Index",
    # Retrieval
    "RetrievalResult",
    "Retriever",
    # Generation
    "AnswerGenerator",
    # Legal Corpus
    "LegalDocument",
    "LegalCorpusError",
    "load_legal_document",
    "load_all_legal_documents",
    "get_available_document_ids",
    "chunk_legal_document",
    "index_legal_corpus",
    # Rule Context
    "RuleContextRetriever",
    "RuleContext",
    "ProvisionContext",
    # Frontend Helpers
    "RuleContextPayload",
    "RelatedProvision",
    "ArticleHit",
    "SemanticHit",
    "SearchResult",
    "get_rule_context",
    "get_related_provisions",
    "search_corpus",
]
