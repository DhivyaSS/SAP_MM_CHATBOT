"""
Pydantic schemas for request and response validation.
"""

from typing import List, Optional, Dict, Any
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Application status")
    environment: str = Field(default="development", description="Execution environment")


class SourceCitation(BaseModel):
    document: str = Field(..., description="Document filename")
    page: int = Field(..., description="Page number")
    chunk_index: Optional[int] = Field(None, description="Chunk index within page/doc")
    snippet: Optional[str] = Field(None, description="Relevant excerpt")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Natural language question about SAP MM")


class ChatResponse(BaseModel):
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="Grounded answer from SAP MM material")
    sources: List[SourceCitation] = Field(default_factory=list, description="Document source citations")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="RAG and graph execution trace")


class DocumentChunkSchema(BaseModel):
    id: Optional[int] = None
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    content: str
    similarity: Optional[float] = None


class DocumentInfo(BaseModel):
    document_name: str
    chunk_count: int
    total_pages: Optional[int] = None
    ingested_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int
