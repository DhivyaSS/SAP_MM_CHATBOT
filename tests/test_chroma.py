import os
import shutil
import pytest
from app.services.chroma_store import ChromaVectorStoreService
from app.config import settings

TEST_PERSIST_DIR = "./data/test_chroma"


@pytest.fixture(autouse=True)
def cleanup_test_dir():
    # Use temporary test directory
    original_dir = settings.CHROMA_PERSIST_DIRECTORY
    settings.CHROMA_PERSIST_DIRECTORY = TEST_PERSIST_DIR
    yield
    settings.CHROMA_PERSIST_DIRECTORY = original_dir
    if os.path.exists(TEST_PERSIST_DIR):
        try:
            shutil.rmtree(TEST_PERSIST_DIR)
        except Exception:
            pass


def test_chroma_save_and_retrieve():
    """
    Test saving chunks and querying similarity with ChromaVectorStoreService.
    """
    chunks = [
        {
            "document_id": "doc_123",
            "document_name": "SAP_MM_Procurement.pdf",
            "page_number": 3,
            "chunk_index": 0,
            "content": "Purchase Orders are created using transaction code ME21N in SAP MM."
        },
        {
            "document_id": "doc_123",
            "document_name": "SAP_MM_Procurement.pdf",
            "page_number": 4,
            "chunk_index": 1,
            "content": "Goods Receipt is posted with transaction code MIGO using movement type 101."
        }
    ]
    # Simple 768-dim mock embeddings
    emb_1 = [0.1] * 768
    emb_2 = [0.9] * 768

    saved_count = ChromaVectorStoreService.save_chunks(chunks, [emb_1, emb_2])
    assert saved_count == 2

    # Check document ingested status
    assert ChromaVectorStoreService.is_document_ingested("SAP_MM_Procurement.pdf") is True
    assert ChromaVectorStoreService.is_document_ingested("Non_Existent.pdf") is False

    # Perform similarity search close to emb_1
    query_vector = [0.1] * 768
    results = ChromaVectorStoreService.similarity_search(query_vector, top_k=2)

    assert len(results) > 0
    assert results[0]["document_name"] == "SAP_MM_Procurement.pdf"
    assert "content" in results[0]

    # Test list_documents
    docs = ChromaVectorStoreService.list_documents()
    assert len(docs) == 1
    assert docs[0]["document_name"] == "SAP_MM_Procurement.pdf"
    assert docs[0]["chunk_count"] == 2
