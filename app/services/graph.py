import logging
from typing import Dict, Any, List, Optional, TypedDict
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
from app.config import settings
from app.services.embedding import GeminiEmbeddingService
from app.services.vector_store import PGVectorStoreService

logger = logging.getLogger(__name__)


# 1. State Definition
class AgentState(TypedDict):
    question: str
    refined_question: Optional[str]
    query_attempts: int
    retrieved_chunks: List[Dict[str, Any]]
    context_sufficient: bool
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    debug_info: Dict[str, Any]


RAG_PROMPT_TEMPLATE = """You are an expert SAP MM (Materials Management) Intelligent Knowledge Assistant.
Your primary task is to answer natural-language questions about SAP MM grounded strictly in the retrieved SAP MM study notes below.

=== STRICT GROUNDING RULES ===
1. Prioritize information explicitly found in the uploaded SAP MM documents.
2. Include relevant details, T-Codes (MM01, ME21N, MIGO, MIRO, SPRO, etc.), process flows, and business roles when supported.
3. If the uploaded documents do NOT contain enough information, clearly state:
   "I couldn't find enough information about this topic in the uploaded SAP MM documents."
4. Do NOT invent facts or pretend that unsupported information came from the documents.
5. Do NOT fabricate citations.

=== RETRIEVED SAP MM CONTEXT ===
{context}

=== USER QUESTION ===
{question}

=== ANSWER ===
"""


class RAGGraphWorkflow:
    """
    LangGraph Stateful Agentic Workflow:
    START -> Question Analysis -> Vector Retrieval -> Evaluation -> (Refine Query if needed) -> Answer Generation -> END
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
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("analyze_question", self.analyze_question_node)
        builder.add_node("retrieve_context", self.retrieve_context_node)
        builder.add_node("evaluate_retrieval", self.evaluate_retrieval_node)
        builder.add_node("refine_query", self.refine_query_node)
        builder.add_node("generate_answer", self.generate_answer_node)

        # Set entry point
        builder.set_entry_point("analyze_question")

        # Edges
        builder.add_edge("analyze_question", "retrieve_context")
        builder.add_edge("retrieve_context", "evaluate_retrieval")

        # Conditional edge based on retrieval evaluation
        builder.add_conditional_edges(
            "evaluate_retrieval",
            self.decide_next_step,
            {
                "refine": "refine_query",
                "generate": "generate_answer"
            }
        )

        builder.add_edge("refine_query", "retrieve_context")
        builder.add_edge("generate_answer", END)

        return builder.compile()

    # Node 1: Question Analysis
    def analyze_question_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[LangGraph] Node 1: Analyzing Question -> '{state['question']}'")
        question = state["question"].strip()
        
        # Expand common SAP MM abbreviations for better vector search
        abbreviations = {
            "pr": "purchase requisition PR",
            "po": "purchase order PO",
            "rfq": "request for quotation RFQ",
            "pir": "purchasing info record PIR",
            "sto": "stock transfer order STO",
            "gr": "goods receipt GR",
            "ir": "invoice receipt IR",
            "liv": "logistics invoice verification LIV",
            "p2p": "procure to pay P2P cycle"
        }
        
        words = question.lower().split()
        expanded_terms = [abbreviations.get(w, w) for w in words]
        refined = " ".join(expanded_terms) if refined != question.lower() else None

        return {
            "refined_question": None,
            "query_attempts": 0,
            "debug_info": {"initial_analysis": f"Processed question: {question}"}
        }

    # Node 2: Retrieval
    def retrieve_context_node(self, state: AgentState) -> Dict[str, Any]:
        attempts = state.get("query_attempts", 0) + 1
        search_query = state.get("refined_question") or state["question"]
        
        logger.info(f"[LangGraph] Node 2: Retrieving SAP MM Context (Attempt #{attempts}) for '{search_query}'")
        
        query_vector = self.embedding_service.embed_text(search_query)
        chunks = PGVectorStoreService.similarity_search(query_vector, top_k=settings.TOP_K)

        return {
            "retrieved_chunks": chunks,
            "query_attempts": attempts
        }

    # Node 3: Evaluate Retrieval
    def evaluate_retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        chunks = state.get("retrieved_chunks", [])
        attempts = state.get("query_attempts", 1)
        
        # Sufficient if we have chunks with similarity score >= 0.35
        sufficient = len(chunks) > 0 and any(c.get("similarity", 0) >= 0.35 for c in chunks)
        logger.info(f"[LangGraph] Node 3: Evaluation -> Context Sufficient: {sufficient} (Attempt #{attempts})")

        return {
            "context_sufficient": sufficient
        }

    # Node Router
    def decide_next_step(self, state: AgentState) -> str:
        if state["context_sufficient"] or state["query_attempts"] >= 2:
            return "generate"
        return "refine"

    # Node 4: Refine Query
    def refine_query_node(self, state: AgentState) -> Dict[str, Any]:
        original_q = state["question"]
        logger.info(f"[LangGraph] Node 4: Refining query for retry -> '{original_q}'")
        
        refined = f"SAP MM {original_q} procurement materials management process details"
        return {
            "refined_question": refined
        }

    # Node 5: Generate Answer
    def generate_answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("[LangGraph] Node 5: Generating Grounded Answer")
        chunks = state.get("retrieved_chunks", [])
        question = state["question"]

        # Format context
        if not chunks or not state["context_sufficient"]:
            answer = "I couldn't find enough information about this topic in the uploaded SAP MM documents."
            sources = []
        else:
            formatted_pieces = []
            seen_sources = set()
            sources = []

            for idx, c in enumerate(chunks, 1):
                doc = c.get("document_name", "Unknown")
                page = c.get("page_number", 0)
                formatted_pieces.append(f"[Chunk {idx} | Source: {doc} — Page {page}]\n{c.get('content', '')}")
                
                key = (doc, page)
                if key not in seen_sources:
                    seen_sources.add(key)
                    sources.append({
                        "document": doc,
                        "page": page,
                        "chunk_index": c.get("chunk_index"),
                        "snippet": c.get("content", "")[:150] + "..."
                    })

            context_str = "\n\n---\n\n".join(formatted_pieces)
            prompt_text = self.prompt_template.format(context=context_str, question=question)

            try:
                res = self.llm.invoke(prompt_text)
                answer = res.content.strip()
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                answer = f"Error during Gemini response generation: {str(e)}"

        debug_info = {
            "user_question": question,
            "refined_question": state.get("refined_question"),
            "query_attempts": state.get("query_attempts"),
            "retrieved_chunk_count": len(chunks),
            "top_similarity_score": chunks[0].get("similarity") if chunks else None,
            "context_sufficient": state.get("context_sufficient"),
            "sources_count": len(sources)
        }

        return {
            "answer": answer,
            "sources": sources,
            "debug_info": debug_info
        }

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow given a user question.
        """
        initial_state: AgentState = {
            "question": question,
            "refined_question": None,
            "query_attempts": 0,
            "retrieved_chunks": [],
            "context_sufficient": False,
            "answer": None,
            "sources": [],
            "debug_info": {}
        }
        
        final_state = self.workflow.invoke(initial_state)
        return {
            "question": question,
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "debug_info": final_state["debug_info"]
        }
