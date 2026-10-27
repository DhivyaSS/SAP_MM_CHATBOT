from unittest.mock import MagicMock, patch
from app.services.embedding import GeminiEmbeddingService


@patch("app.services.embedding.GoogleGenerativeAIEmbeddings")
def test_gemini_embedding_mock(mock_google_embeddings):
    """
    Test GeminiEmbeddingService using a mock to avoid live API calls.
    """
    mock_instance = MagicMock()
    mock_instance.embed_query.return_value = [0.1] * 768
    mock_instance.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
    mock_google_embeddings.return_value = mock_instance

    service = GeminiEmbeddingService(api_key="fake_key_for_testing")
    
    # Test query embedding
    vec = service.embed_text("What is a purchase requisition?")
    assert len(vec) == 768
    assert vec[0] == 0.1

    # Test doc batch embedding
    batch_vecs = service.embed_documents(["chunk text 1", "chunk text 2"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 768
