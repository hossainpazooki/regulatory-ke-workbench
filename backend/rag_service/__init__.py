"""RAG service - Retrieval and generation for Q&A."""

from .app.services.retriever import Retriever
from .app.services.bm25 import BM25Index
from .app.services.corpus_loader import load_legal_document, get_available_document_ids

__all__ = [
    "Retriever",
    "BM25Index",
    "load_legal_document",
    "get_available_document_ids",
]
