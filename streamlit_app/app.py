import streamlit as st
import requests
import json
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page Config
st.set_page_config(
    page_title="SAP MM Intelligent Knowledge Assistant",
    page_icon="📦",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F3F4F6;
        border-left: 4px solid #2563EB;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
    .citation-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


def get_ingested_documents():
    try:
        res = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"documents": [], "total_chunks": 0}


# Header
st.markdown('<div class="main-title">📦 SAP MM Intelligent Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Grounded RAG Assistant powered by Google Gemini, ChromaDB / pgvector, LangChain & LangGraph</div>', unsafe_allow_html=True)

# System Health Check
backend_online = check_backend_health()
if not backend_online:
    st.warning("⚠️ FastAPI Backend is offline or starting up. Ensure server is running on `http://localhost:8000`.")

# Sidebar - Document Management
with st.sidebar:
    st.header("📄 Document Management")
    st.write("Upload your SAP MM learning notes, procurement guides, and study material.")

    uploaded_file = st.file_uploader("Upload SAP MM PDF Document", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
                    if res.status_code == 201:
                        data = res.json()
                        st.success(f"Successfully ingested **{data['document_name']}** ({data['pages_extracted']} pages, {data['chunks_indexed']} chunks).")
                        st.rerun()
                    elif res.status_code == 409:
                        st.info(f"ℹ️ Document **{uploaded_file.name}** is already ingested.")
                    else:
                        st.error(f"Ingestion failed: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to server: {e}")

    st.markdown("---")
    st.subheader("📚 Indexed Knowledge Base")
    doc_data = get_ingested_documents()
    docs = doc_data.get("documents", [])
    total_chunks = doc_data.get("total_chunks", 0)

    st.metric(label="Total Indexed Chunks", value=total_chunks)

    if docs:
        for d in docs:
            st.markdown(f"• **{d['document_name']}** ({d['chunk_count']} chunks, {d['total_pages']} pages)")
    else:
        st.info("No SAP MM documents uploaded yet. Upload a PDF above to get started.")

# Main Interface Tabs
tab1, tab2 = st.tabs(["💬 Ask SAP MM", "🔍 RAG Execution Debugger"])

with tab1:
    st.markdown("### Sample Questions")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    
    selected_sample = None
    if sample_col1.button("What is a Purchase Requisition?"):
        selected_sample = "What is a purchase requisition?"
    if sample_col2.button("Difference between PR & PO?"):
        selected_sample = "What is the difference between a purchase requisition and a purchase order?"
    if sample_col3.button("Explain Source List"):
        selected_sample = "What is a source list?"
    if sample_col4.button("Explain Info Record"):
        selected_sample = "What is the purpose of an info record?"

    user_query = st.text_input(
        "Ask a question about your SAP MM documents:",
        value=selected_sample if selected_sample else "",
        placeholder="e.g. Explain the standard procurement process in SAP MM..."
    )

    ask_button = st.button("Ask Assistant", type="primary")

    if (ask_button or selected_sample) and user_query:
        with st.spinner("Retrieving SAP MM chunks & generating grounded answer..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={"question": user_query},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["last_result"] = result
                else:
                    st.error(f"Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        
        st.markdown("---")
        st.markdown("### Answer")
        st.markdown(res["answer"])

        st.markdown("### Sources")
        sources = res.get("sources", [])
        if sources:
            for s in sources:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <span class="citation-badge">📄 {s['document']} — Page {s['page']}</span>
                        <div style="margin-top: 0.4rem; font-size: 0.9rem; color: #374151;">
                            <em>"{s.get('snippet', '')}"</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("No specific source citations matched for this question.")

with tab2:
    st.markdown("### 🔍 RAG & LangGraph Execution Trace")
    st.write("This debug view exposes the internal RAG vector retrieval scores and graph state for educational inspection.")

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        debug = res.get("debug_info", {})

        st.json(debug)
    else:
        st.info("Ask a question in the 'Ask SAP MM' tab to inspect the RAG retrieval pipeline.")
