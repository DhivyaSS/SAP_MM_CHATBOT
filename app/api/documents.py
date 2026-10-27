import logging
import uuid
from typing import List
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.models.schemas import DocumentListResponse, DocumentInfo
from app.services.pdf_loader import PDFLoaderService
from app.services.chunker import DocumentChunkerService
from app.services.embedding import GeminiEmbeddingService
from app.services.vector_store import PGVectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

chunker_service = DocumentChunkerService()
embedding_service = GeminiEmbeddingService()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_and_ingest_document(file: UploadFile = File(...)):
    """
    POST /documents/upload
    Accepts PDF file upload, extracts pages, chunks text, generates Gemini embeddings,
    and stores chunks in PostgreSQL pgvector. Prevents duplicate ingestion.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported."
        )

    # Check for duplicate document ingestion
    if PGVectorStoreService.is_document_ingested(file.filename):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document '{file.filename}' has already been ingested into the vector database."
        )

    try:
        content_bytes = await file.read()
        if len(content_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF file is empty."
            )

        doc_id = str(uuid.uuid4())
        filename = file.filename

        # 1. Extract text page-by-page
        pages = PDFLoaderService.extract_pages_from_bytes(content_bytes, filename)
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not extract printable text from PDF '{filename}'."
            )

        # 2. Chunk text
        chunks = chunker_service.chunk_pages(pages, document_id=doc_id)

        # 3. Generate embeddings
        chunk_texts = [c["content"] for c in chunks]
        embeddings = embedding_service.embed_documents(chunk_texts)

        # 4. Save to pgvector
        saved_count = PGVectorStoreService.save_chunks(chunks, embeddings)

        return {
            "status": "success",
            "message": f"Successfully ingested PDF document '{filename}'.",
            "document_id": doc_id,
            "document_name": filename,
            "pages_extracted": len(pages),
            "chunks_indexed": saved_count
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to ingest document '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """
    GET /documents
    Lists all ingested documents, total page counts, chunk totals, and ingestion timestamps.
    """
    try:
        raw_docs = PGVectorStoreService.list_documents()
        doc_models = [
            DocumentInfo(
                document_name=d["document_name"],
                chunk_count=d["chunk_count"],
                total_pages=d["total_pages"],
                ingested_at=d["ingested_at"]
            )
            for d in raw_docs
        ]
        total_chunks = sum(d["chunk_count"] for d in raw_docs)
        return DocumentListResponse(documents=doc_models, total_chunks=total_chunks)
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve document list: {str(e)}"
        )
