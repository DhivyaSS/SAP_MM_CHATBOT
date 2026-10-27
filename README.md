# 📦 SAP MM Intelligent Knowledge Assistant

An end-to-end, production-style GenAI & RAG portfolio project built with **Python**, **FastAPI**, **ChromaDB / PostgreSQL + pgvector**, **LangChain**, **LangGraph**, **Google Gemini API**, and **Streamlit**.

The application allows users to upload SAP Materials Management (MM) learning notes, study materials, and PDFs, and ask natural-language questions grounded strictly in the uploaded documents.

---

## 📌 Problem & Core Use Case

SAP Materials Management (MM) is a complex ERP module spanning organizational structures, master data (Material Master, Vendor Master, PIR, Source List, Quota Arrangement), procurement cycles (PR, RFQ, PO, STO, Subcontracting, Consignment), inventory management (MIGO, Movement Types 101, 201, 301, 561), and logistics invoice verification (MIRO, 3-Way Match).

General-purpose LLMs often hallucinate or confuse custom enterprise configurations. The **SAP MM Intelligent Knowledge Assistant** uses **Retrieval-Augmented Generation (RAG)** to:
1. Ground answers in exact user-uploaded PDF notes.
2. Provide precise document and page-level citations (e.g. `SAP_MM_Notes.pdf — Page 8`).
3. Explicitly decline to answer when documents lack sufficient information (*"I couldn't find enough information about this topic in the uploaded SAP MM documents."*).

---

## 🏗️ Architecture

```
                    SAP MM PDF DOCUMENTS
                             │
                             ▼
                      Document Loader (pypdf)
                             │
                             ▼
                 Recursive Text Chunking (1000 chars, 200 overlap)
                             │
                             ▼
                Gemini Embeddings (models/text-embedding-004)
                             │
                             ▼
                   PostgreSQL + pgvector (768-dim HNSW Cosine Index)
                             │
                             │
                             ▼
USER → Streamlit UI → FastAPI Backend → LangGraph Agentic Workflow
                                                 │
                                                 ▼
                                             LangChain
                                                 │
                                                 ▼
                                           RAG Retriever
                                                 │
                                                 ▼
                                       PostgreSQL / pgvector
                                                 │
                                                 ▼
                                          Relevant Chunks
                                                 │
                                                 ▼
                                       Grounded Prompt
                                                 │
                                                 ▼
                                            Gemini LLM
                                                 │
                                                 ▼
                                       Answer + Citations
```

---

## 🔄 RAG Pipeline & Agentic Workflow

### 1. Ingestion Pipeline
- **Document Loading**: `PDFLoaderService` extracts text page-by-page from raw PDF bytes.
- **Chunking**: `DocumentChunkerService` splits text into overlapping chunks using `RecursiveCharacterTextSplitter`, preserving `document_name`, `page_number`, and `chunk_index`.
- **Embeddings**: `GeminiEmbeddingService` computes 768-dimensional dense vectors using Google Gemini `models/text-embedding-004`.
- **Vector Storage**: `PGVectorStoreService` persists chunks and vectors into PostgreSQL using the `pgvector` extension and HNSW cosine similarity index.

### 2. Retrieval & Generation Pipeline
- **Query Embedding**: Converts user question into a 768-dim query vector.
- **pgvector Cosine Search**: Executes distance query `(embedding <=> query_vector)` in PostgreSQL to retrieve the top-k (`4`) most relevant chunks.
- **LangChain Integration**: Constructs a grounded prompt template enforcing strict zero-hallucination rules and SAP terminology preservation.
- **LangGraph Agentic StateGraph**:
  - **Node 1 (`analyze_question`)**: Analyzes question and expands SAP MM abbreviations (e.g., PR -> Purchase Requisition, PO -> Purchase Order).
  - **Node 2 (`retrieve_context`)**: Performs vector similarity search.
  - **Node 3 (`evaluate_retrieval`)**: Evaluates context sufficiency score.
  - **Node 4 (`refine_query`)**: If relevance is low, expands query terms and retries retrieval.
  - **Node 5 (`generate_answer`)**: Invokes Gemini LLM (`gemini-1.5-flash`) to generate grounded response + citations.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, `pydantic-settings`
- **Database & Vector Store**: PostgreSQL 16 + `pgvector`, HNSW Indexing
- **LLM & Embeddings**: Google Gemini API (`gemini-1.5-flash`, `models/text-embedding-004`)
- **Orchestration**: LangChain, LangGraph StateGraph
- **PDF Processing**: `pypdf`
- **Frontend**: Streamlit
- **Testing**: Pytest, FastAPI TestClient, unittest.mock
- **Containerization**: Docker, Docker Compose

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Application operational health check |
| `POST` | `/documents/upload` | Upload & ingest PDF document into pgvector |
| `GET` | `/documents` | List all ingested documents and chunk totals |
| `POST` | `/chat` | Submit SAP MM question to LangGraph RAG workflow |

---

## 💻 Local Setup & Execution Guide

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Google Gemini API Key ([Get one free at Google AI Studio](https://aistudio.google.com/))

### Step 1: Clone & Configure Environment
```bash
cp .env.example .env
```
Edit `.env` and set your `GEMINI_API_KEY`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### Step 2: Run Locally with ChromaDB (Recommended — No Docker Required!)
By default, the project is configured to use local **ChromaDB** (`VECTOR_STORE_TYPE=chroma`), storing vectors on disk in `./data/chroma`.

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI backend server:
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```
3. In a second terminal, start the Streamlit UI:
   ```bash
   streamlit run streamlit_app/app.py
   ```
Access the application at:
- **Streamlit Frontend**: `http://localhost:8501`
- **FastAPI Documentation**: `http://localhost:8000/docs`

---

### Step 3: Optional — Run with Docker & PostgreSQL + pgvector
If you prefer running with PostgreSQL and pgvector via Docker:
1. In `.env`, set:
   ```env
   VECTOR_STORE_TYPE=pgvector
   ```
2. Run Docker Compose:
   ```bash
   docker compose up --build
   ```

---

## 🧪 Running Unit Tests

Run the complete test suite using `pytest`:
```bash
pytest -v
```

---

## ❓ Example SAP MM Questions

- *"What is a purchase requisition?"*
- *"What is the difference between a purchase requisition and a purchase order?"*
- *"Explain the standard procurement process."*
- *"What is a source list?"*
- *"What is the purpose of an info record?"*
- *"Explain release orders and release strategies in SAP MM."*
- *"What are the movement types for Goods Receipt and Goods Issue?"*

---

## 🔮 Future Improvements

- **Reranking**: Integrate Cohere or BGE Cross-Encoder rerankers to improve top-k context precision.
- **Hybrid Search**: Combine pgvector dense vector search with PostgreSQL `tsvector` full-text keyword search (BM25).
- **RAG Evaluation**: Implement Ragas or TruLens evaluation pipelines measuring Faithfulness, Answer Relevance, and Context Recall.
- **LangSmith Tracing**: Integrate LangSmith for full tracing of agentic graph executions.
- **Structured Knowledge Graph**: Graph RAG linking SAP entities (Plants, Storage Locations, Purchasing Orgs, Vendors, Materials).
