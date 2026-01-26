"""Tests for RAG integration with legal corpus."""

import pytest

from backend.rag import (
    BM25Index,
    Retriever,
    load_legal_document,
    load_all_legal_documents,
    chunk_legal_document,
    index_legal_corpus,
)


class TestChunkLegalDocument:
    """Tests for chunking legal documents."""

    def test_chunks_mica_document(self):
        """Test chunking MiCA document."""
        doc = load_legal_document("mica_2023")
        chunks = chunk_legal_document(doc)

        assert len(chunks) > 0
        # Each chunk should have required fields
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["document_id"] == "mica_2023"
            assert chunk["metadata"]["source_type"] == "legal_text"

    def test_chunks_have_article_refs(self):
        """Test that chunks have article references where possible."""
        doc = load_legal_document("mica_2023")
        chunks = chunk_legal_document(doc)

        # At least some chunks should have article references
        chunks_with_articles = [
            c for c in chunks if c["metadata"].get("article")
        ]
        assert len(chunks_with_articles) > 0

    def test_chunks_have_document_metadata(self):
        """Test that chunks have document metadata."""
        doc = load_legal_document("mica_2023")
        chunks = chunk_legal_document(doc)

        for chunk in chunks:
            meta = chunk["metadata"]
            assert meta.get("document_title")
            assert meta.get("citation")
            assert meta.get("jurisdiction") == "EU"

    def test_chunk_ids_are_unique(self):
        """Test that chunk IDs are unique."""
        doc = load_legal_document("mica_2023")
        chunks = chunk_legal_document(doc)

        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


class TestIndexLegalCorpus:
    """Tests for indexing legal corpus into BM25."""

    def test_index_legal_corpus_returns_count(self):
        """Test that indexing returns chunk count."""
        index = BM25Index()
        count = index_legal_corpus(index)

        assert count > 0
        assert len(index) == count

    def test_index_legal_corpus_with_retriever(self):
        """Test indexing with Retriever (no vectors)."""
        retriever = Retriever(use_vectors=False)
        count = index_legal_corpus(retriever)

        assert count > 0
        # Retriever may further chunk documents, so len >= count
        assert len(retriever) >= count


class TestSearchLegalCorpus:
    """Tests for searching indexed legal corpus."""

    @pytest.fixture
    def indexed_bm25(self):
        """Create a BM25 index with legal corpus."""
        index = BM25Index()
        index_legal_corpus(index)
        return index

    def test_search_mica_keyword(self, indexed_bm25):
        """Test searching for MiCA-specific keyword."""
        results = indexed_bm25.search("asset-referenced tokens", top_k=5)

        assert len(results) > 0
        # At least one result should be from MiCA
        mica_results = [
            r for r, score in results
            if r.metadata.get("document_id") == "mica_2023"
        ]
        assert len(mica_results) > 0

    def test_search_dlt_keyword(self, indexed_bm25):
        """Test searching for DLT Pilot-specific keyword."""
        results = indexed_bm25.search("DLT market infrastructure", top_k=5)

        assert len(results) > 0
        # At least one result should be from DLT Pilot
        dlt_results = [
            r for r, score in results
            if r.metadata.get("document_id") == "dlt_pilot_2022"
        ]
        assert len(dlt_results) > 0

    def test_search_genius_keyword(self, indexed_bm25):
        """Test searching for GENIUS Act-specific keyword."""
        results = indexed_bm25.search("stablecoin reserves", top_k=5)

        assert len(results) > 0
        # At least one result should be from GENIUS
        genius_results = [
            r for r, score in results
            if r.metadata.get("document_id") == "genius_act_2025"
        ]
        assert len(genius_results) > 0

    def test_search_results_have_metadata(self, indexed_bm25):
        """Test that search results include metadata."""
        results = indexed_bm25.search("authorization", top_k=5)

        for doc, score in results:
            assert doc.metadata.get("source_type") == "legal_text"
            assert doc.metadata.get("document_id")

    def test_search_with_retriever(self):
        """Test searching using Retriever interface."""
        retriever = Retriever(use_vectors=False)
        index_legal_corpus(retriever)

        results = retriever.search("crypto-asset service providers", top_k=5)

        assert len(results) > 0
        for result in results:
            assert result.text
            assert result.document_id
            assert result.metadata.get("source_type") == "legal_text"


class TestLegalCorpusCoverage:
    """Tests for legal corpus coverage."""

    @pytest.fixture
    def indexed_bm25(self):
        """Create a BM25 index with legal corpus."""
        index = BM25Index()
        index_legal_corpus(index)
        return index

    def test_all_three_documents_indexed(self, indexed_bm25):
        """Test that all three documents are in the index."""
        # Search for something generic
        results = indexed_bm25.search("shall", top_k=50)

        doc_ids = {r.metadata.get("document_id") for r, _ in results}

        assert "mica_2023" in doc_ids
        assert "dlt_pilot_2022" in doc_ids
        assert "genius_act_2025" in doc_ids

    def test_eu_jurisdiction_filter(self, indexed_bm25):
        """Test that EU documents are correctly tagged."""
        results = indexed_bm25.search("Member State", top_k=10)

        for doc, score in results:
            if doc.metadata.get("document_id") in ("mica_2023", "dlt_pilot_2022"):
                assert doc.metadata.get("jurisdiction") == "EU"

    def test_us_jurisdiction_filter(self, indexed_bm25):
        """Test that US documents are correctly tagged."""
        results = indexed_bm25.search("Federal Reserve", top_k=10)

        for doc, score in results:
            if doc.metadata.get("document_id") == "genius_act_2025":
                assert doc.metadata.get("jurisdiction") == "US"


class TestArticleExtraction:
    """Tests for article reference extraction in chunks."""

    def test_article_numbers_extracted(self):
        """Test that article numbers are extracted from MiCA."""
        doc = load_legal_document("mica_2023")
        chunks = chunk_legal_document(doc)

        # Find chunks with Article 36
        art36_chunks = [
            c for c in chunks
            if c["metadata"].get("article") and "36" in str(c["metadata"]["article"])
        ]
        # Should find at least one chunk referencing Article 36
        assert len(art36_chunks) > 0

    def test_section_numbers_extracted(self):
        """Test that section numbers are extracted from GENIUS Act."""
        doc = load_legal_document("genius_act_2025")
        chunks = chunk_legal_document(doc)

        # Find chunks with Section numbers
        section_chunks = [
            c for c in chunks
            if c["metadata"].get("article")
        ]
        # Should find sections
        assert len(section_chunks) > 0
