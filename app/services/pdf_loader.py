import logging
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from pypdf import PdfReader
from io import BytesIO

logger = logging.getLogger(__name__)


class PDFLoaderService:
    """
    Extracts text page-by-page from uploaded PDF documents while preserving page numbers.
    """

    @staticmethod
    def extract_pages_from_bytes(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract text from raw PDF bytes.
        Returns a list of dicts: [{'page_number': 1, 'text': '...'}, ...]
        """
        pages = []
        try:
            reader = PdfReader(BytesIO(file_bytes))
            total_pages = len(reader.pages)
            logger.info(f"Processing PDF '{filename}' with {total_pages} pages.")

            for page_idx, page in enumerate(reader.pages):
                page_number = page_idx + 1
                text = page.extract_text() or ""
                cleaned_text = text.strip()
                if cleaned_text:
                    pages.append({
                        "document_name": filename,
                        "page_number": page_number,
                        "content": cleaned_text
                    })
            logger.info(f"Successfully extracted {len(pages)} non-empty pages from '{filename}'.")
            return pages
        except Exception as e:
            logger.error(f"Failed to extract text from PDF '{filename}': {e}")
            raise ValueError(f"Could not parse PDF file '{filename}': {str(e)}")
