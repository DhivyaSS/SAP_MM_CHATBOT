import logging
from typing import List, Dict, Any, Tuple
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
from app.config import settings
from app.services.embedding import GeminiEmbeddingService
from app.services.vector_store import PGVectorStoreService

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are an expert SAP MM (Materials Management) Intelligent Knowledge Assistant.
Your primary task is to answer natural-language questions about SAP MM grounded strictly in the retrieved SAP MM study notes and documents below.

=== STRICT GROUNDING RULES ===
1. Prioritize information explicitly found in the uploaded SAP MM documents.
2. If the context contains relevant SAP MM information (e.g., T-Codes like MM01, ME21N, MIGO, MIRO, SPRO, process flows like P2P/Procure-to-Pay, master data like Material/Vendor Master, PIR, Source List, STO), explain it clearly using proper SAP terminology.
3. If the uploaded documents do NOT contain enough information to answer the question, clearly state:
   "I couldn't find enough information about this topic in the uploaded SAP MM documents."
4. Do NOT fabricate information or pretend that unsupported information came from the documents.
5. Do NOT invent fake document names or page citations.
6. Distinguish clearly between:
   - Facts explicitly documented in the uploaded notes.
   - Reasonable explanations or inferences.
   - Information not available in the uploaded documents.

=== RETRIEVED SAP MM CONTEXT ===
{context}

=== USER QUESTION ===
{question}

=== ANSWER ===
"""


class RAGChainService:
    """
    Orchestrates the LangChain RAG pipeline:
    Question -> Embedding -> PGVector Retrieval -> Grounded Prompt Construction -> Gemini LLM -> Answer + Citations
    """

    def __init__(self):
        self.embedding_service = GeminiEmbeddingService()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1
        )
        self.prompt_template = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def retrieve_context(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Embed question and perform similarity search against pgvector.
        """
        query_vector = self.embedding_service.embed_text(question)
        retrieved_chunks = PGVectorStoreService.similarity_search(query_vector, top_k=top_k)
        return retrieved_chunks

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a clean context string for the prompt.
        """
        if not chunks:
            return "No relevant SAP MM context found in the uploaded documents."

        formatted_pieces = []
        for idx, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("document_name", "Unknown Document")
            page_num = chunk.get("page_number", "N/A")
            chunk_idx = chunk.get("chunk_index", "N/A")
            content = chunk.get("content", "")
            
            piece = f"[Chunk {idx} | Source: {doc_name} — Page {page_num} (Chunk #{chunk_idx})]\n{content}"
            formatted_pieces.append(piece)

        return "\n\n---\n\n".join(formatted_pieces)

    def extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique document/page citations from retrieved chunks.
        """
        seen = set()
        citations = []
        for chunk in chunks:
            doc = chunk.get("document_name")
            page = chunk.get("page_number")
            key = (doc, page)
            if doc and page and key not in seen:
                seen.add(key)
                citations.append({
                    "document": doc,
                    "page": page,
                    "chunk_index": chunk.get("chunk_index"),
                    "snippet": chunk.get("content", "")[:150] + "..."
                })
        return citations

    def run_rag(self, question: str, top_k: int = None) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute full RAG pipeline:
        Retrieval -> Context Formatting -> Gemini Call -> Response & Citations
        """
        # 1. Retrieve relevant chunks
        chunks = self.retrieve_context(question, top_k=top_k)
        
        # 2. Format context for prompt
        formatted_context = self.format_context(chunks)
        
        # 3. Extract source citations
        citations = self.extract_citations(chunks)

        # 4. Check if context is completely empty
        if not chunks or all(c.get("similarity", 0) < 0.3 for c in chunks):
            insufficient_msg = "I couldn't find enough information about this topic in the uploaded SAP MM documents."
            debug_info = {
                "question": question,
                "retrieved_chunk_count": len(chunks),
                "formatted_context": formatted_context,
                "status": "insufficient_context"
            }
            return insufficient_msg, [], debug_info

        # 5. Build prompt & invoke Gemini LLM
        prompt_text = self.prompt_template.format(
            context=formatted_context,
            question=question
        )
        
        try:
            response = self.llm.invoke(prompt_text)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"Error invoking Gemini LLM in RAG pipeline: {e}")
            answer = f"An error occurred while contacting the Gemini API: {str(e)}"

        debug_info = {
            "question": question,
            "retrieved_chunk_count": len(chunks),
            "formatted_context": formatted_context,
            "prompt_used": prompt_text,
            "status": "success"
        }

        return answer, citations, debug_info
