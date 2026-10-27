from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_chat_validation_empty_question():
    """
    Test POST /chat returns HTTP 400 or 422 if question is empty or missing.
    """
    response = client.post("/chat", json={"question": ""})
    assert response.status_code in (400, 422)


def test_document_upload_non_pdf():
    """
    Test POST /documents/upload rejects non-PDF file formats.
    """
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", b"plain text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


@patch("app.api.chat.graph_workflow.run")
def test_chat_endpoint_success(mock_graph_run):
    """
    Test POST /chat endpoint returns answer and citations when mocked.
    """
    mock_graph_run.return_value = {
        "question": "What is a purchase requisition?",
        "answer": "A Purchase Requisition (PR) is an internal document used to request materials.",
        "sources": [
            {"document": "SAP_MM_Notes.pdf", "page": 8, "chunk_index": 1, "snippet": "PR notes..."}
        ],
        "debug_info": {"status": "success"}
    }

    response = client.post("/chat", json={"question": "What is a purchase requisition?"})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is a purchase requisition?"
    assert "Purchase Requisition" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document"] == "SAP_MM_Notes.pdf"
    assert data["sources"][0]["page"] == 8
