from app.services.chunker import DocumentChunkerService


def test_chunker_page_metadata_preservation():
    """
    Test DocumentChunkerService preserves document_name, page_number, and chunk_index.
    """
    chunker = DocumentChunkerService(chunk_size=100, chunk_overlap=20)
    
    sample_pages = [
        {
            "document_name": "SAP_MM_PR_Notes.pdf",
            "page_number": 8,
            "content": "A Purchase Requisition (PR) is an internal document created in SAP MM to request purchasing department to procure materials or services."
        },
        {
            "document_name": "SAP_MM_PR_Notes.pdf",
            "page_number": 9,
            "content": "Standard T-Code for Purchase Requisition is ME51N. PR can be created manually or automatically through MRP."
        }
    ]
    
    chunks = chunker.chunk_pages(sample_pages, document_id="doc-uuid-123")
    
    assert len(chunks) >= 2
    assert chunks[0]["document_name"] == "SAP_MM_PR_Notes.pdf"
    assert chunks[0]["page_number"] == 8
    assert chunks[0]["chunk_index"] == 1
    assert "Purchase Requisition" in chunks[0]["content"]
    assert chunks[-1]["page_number"] == 9
