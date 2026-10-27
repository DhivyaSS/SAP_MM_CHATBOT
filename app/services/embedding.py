import logging
from typing import List
# pyrefly: ignore [missing-import]
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingService:
    """
    Dedicated embedding service communicating with Google Gemini API
    to compute 768-dimensional dense vector embeddings.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model or settings.EMBEDDING_MODEL
        
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY is missing or invalid. Embeddings won't work until configured.")

        # Initialize LangChain Google Generative AI Embeddings
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key
        )

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single query text.
        """
        try:
            vector = self._embeddings.embed_query(text)
            return vector
        except Exception as e:
            logger.error(f"Error generating query embedding via Gemini: {e}")
            raise RuntimeError(f"Gemini Embedding API call failed: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of text chunks.
        """
        try:
            vectors = self._embeddings.embed_documents(texts)
            return vectors
        except Exception as e:
            logger.error(f"Error generating document batch embeddings via Gemini: {e}")
            raise RuntimeError(f"Gemini Batch Embedding API call failed: {str(e)}")
