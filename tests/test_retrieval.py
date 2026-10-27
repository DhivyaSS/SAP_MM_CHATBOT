from unittest.mock import patch, MagicMock
from app.services.rag_chain import RAGChainService


def test_rag_citation_extraction():
    """
    Test extraction of unique document citations from retrieved chunks.
    """
    rag_service = RAGChainService()
    sample_chunks = [
        {"document_name": "SAP_MM_Notes.pdf", "page_number": 8, "chunk_index": 1, "content": "Sample content page 8 chunk 1"},
        {"document_name": "SAP_MM_Notes.pdf", "page_number": 8, "chunk_index": 2, "content": "Sample content page 8 chunk 2"},
        {"document_name": "Procurement.pdf", "page_number": 14, "chunk_index": 5, "content": "Sample content page 14"}
    ]
    
    citations = rag_service.extract_citations(sample_chunks)
    
    # Should extract 2 unique (doc, page) citations
    assert len(citations) == 2
    assert citations[0]["document"] == "SAP_MM_Notes.pdf"
    assert citations[0]["page"] == 8
    assert citations[1]["document"] == "Procurement.pdf"
    assert citations[1]["page"] == 14


@patch("app.services.rag_chain.PGVectorStoreService.similarity_search")
@patch("app.services.rag_chain.GeminiEmbeddingService.embed_text")
def test_insufficient_context_fallback(mock_embed, mock_search):
    """
    Test fallback message when no relevant chunks are found.
    """
    mock_embed.return_value = [0.0] * 768
    mock_search.return_value = [] # Empty retrieval

    rag_service = RAGChainService()
    answer, citations, debug = rag_service.run_rag("What is a quantum computer?")
    
    assert "couldn't find enough information" in answer.lower()
    assert len(citations) == 0
    assert debug["status"] == "insufficient_context"
