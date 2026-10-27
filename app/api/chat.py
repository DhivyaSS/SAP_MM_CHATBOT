import logging
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import ChatRequest, ChatResponse
from app.services.graph import RAGGraphWorkflow

logger = logging.getLogger(__name__)
router = APIRouter()

# Instantiate LangGraph agentic workflow
graph_workflow = RAGGraphWorkflow()


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    POST /chat endpoint executing the stateful LangGraph agentic RAG workflow.
    Returns grounded SAP MM answer, document citations, and retrieval debug trace.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty."
        )

    try:
        logger.info(f"Received chat request: '{request.question}'")
        result = graph_workflow.run(request.question)
        
        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            debug_info=result["debug_info"]
        )
    except Exception as e:
        logger.error(f"Error executing chat workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your question: {str(e)}"
        )
