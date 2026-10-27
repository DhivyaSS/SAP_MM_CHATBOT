import logging
from typing import List, Dict, Any
try:
    # pyrefly: ignore [missing-import]
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    # pyrefly: ignore [missing-import]
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings

logger = logging.getLogger(__name__)


class DocumentChunkerService:
    """
    Splits extracted page text into smaller chunks with overlap while preserving document metadata.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_pages(self, pages: List[Dict[str, Any]], document_id: str) -> List[Dict[str, Any]]:
        """
        Takes page objects [{'document_name': '...', 'page_number': 1, 'content': '...'}]
        and produces chunk objects retaining metadata and index.
        """
        all_chunks = []
        global_chunk_index = 0

        for page in pages:
            doc_name = page["document_name"]
            page_num = page["page_number"]
            page_text = page["content"]

            # Split text using LangChain's RecursiveCharacterTextSplitter
            text_splits = self.text_splitter.split_text(page_text)

            for split_text in text_splits:
                if split_text.strip():
                    global_chunk_index += 1
                    all_chunks.append({
                        "document_id": document_id,
                        "document_name": doc_name,
                        "page_number": page_num,
                        "chunk_index": global_chunk_index,
                        "content": split_text.strip(),
                        "metadata": {
                            "document_name": doc_name,
                            "page_number": page_num,
                            "chunk_index": global_chunk_index
                        }
                    })

        logger.info(f"Chunked document '{document_id}' into {len(all_chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap}).")
        return all_chunks
