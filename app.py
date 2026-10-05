"""
================================================================================
Intelligent Legal Document Answering
BTech Capstone / College AI Project
--------------------------------------------------------------------------------
Architecture: Retrieval-Augmented Generation (RAG)
Pipeline: Legal PDF -> PyPDF -> Text Cleaning -> Chunking ->
          SentenceTransformers (all-MiniLM-L6-v2) -> ChromaDB ->
          Query Retrieval -> LLM (via requests) -> Answer & Page-Numbered Sources
================================================================================
"""

import os
import sys
import types
import importlib.machinery
from unittest.mock import MagicMock

# -----------------------------------------------------------------------------
# Windows AppLocker / WDAC Environment Compatibility Shim
# Prevents unused sklearn evaluation modules from triggering OpenMP DLL load blocks
# -----------------------------------------------------------------------------
class AutoMockModule(types.ModuleType):
    def __getattr__(self, name):
        m = MagicMock()
        setattr(self, name, m)
        return m

if 'sklearn' not in sys.modules:
    mock_sklearn = AutoMockModule('sklearn')
    mock_sklearn.__spec__ = importlib.machinery.ModuleSpec('sklearn', None)
    mock_sklearn.__version__ = '1.5.0'
    mock_sklearn.__path__ = []
    sys.modules['sklearn'] = mock_sklearn

    for sub in ['metrics', 'metrics.pairwise', 'utils', 'utils._openmp_helpers', 'base', 'preprocessing', 'feature_extraction', 'linear_model', 'cluster', 'neighbors']:
        m = AutoMockModule('sklearn.' + sub)
        m.__spec__ = importlib.machinery.ModuleSpec('sklearn.' + sub, None)
        m.__path__ = []
        sys.modules['sklearn.' + sub] = m
        setattr(mock_sklearn, sub.split('.')[-1], m)

import re
import time
import json
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
import requests
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 1. Environment & Setup
# -----------------------------------------------------------------------------
load_dotenv(override=True)

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "legal_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# -----------------------------------------------------------------------------
# 2. Pydantic Models for Structured Data
# -----------------------------------------------------------------------------
class DocumentChunk(BaseModel):
    """Pydantic model representing a single document chunk with page metadata."""
    id: str = Field(description="Unique identifier for the chunk")
    text: str = Field(description="Extracted and cleaned chunk text content")
    page: int = Field(description="1-indexed source document page number")
    chunk_number: int = Field(description="Sequential chunk number within document")
    source: str = Field(description="Filename of the source document")

class RetrievedSource(BaseModel):
    """Pydantic model representing a retrieved context chunk."""
    id: str
    text: str
    page: int
    chunk: int
    source: str
    distance: float

# -----------------------------------------------------------------------------
# 3. Streamlit Page Configuration & Modern Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Legal Document Answering",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern legal-tech design aesthetic
st.markdown("""
<style>
    /* Global Fonts & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Header Styling */
    .legal-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .dark .legal-title, [data-theme="dark"] .legal-title {
        color: #F8FAFC;
    }
    .legal-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    
    /* Legal Safety Notice Card */
    .legal-disclaimer {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(245, 158, 11, 0.08));
        border-left: 4px solid #F59E0B;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #B45309;
        margin-bottom: 1.5rem;
        line-height: 1.45;
    }
    .dark .legal-disclaimer, [data-theme="dark"] .legal-disclaimer {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(245, 158, 11, 0.15));
        color: #FCD34D;
        border-left-color: #F59E0B;
    }
    
    /* Stats & Metric Badges */
    .stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .dark .stat-card, [data-theme="dark"] .stat-card {
        background: #1E293B;
        border-color: #334155;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0284C7;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* RAG Architecture Chip */
    .rag-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        background-color: #E0F2FE;
        color: #0369A1;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-bottom: 0.8rem;
    }
    .dark .rag-badge, [data-theme="dark"] .rag-badge {
        background-color: #0C4A6E;
        color: #7DD3FC;
    }

    /* Source Box Styling */
    .source-box {
        background: #F1F5F9;
        border-radius: 6px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-family: monospace;
        font-size: 0.84rem;
        white-space: pre-wrap;
    }
    .dark .source-box, [data-theme="dark"] .source-box {
        background: #0F172A;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Cached Model and Vector Database Initializers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer embedding model.
    Using all-MiniLM-L6-v2: Fast, high accuracy, produces 384-dimensional embeddings.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_chroma_client() -> chromadb.ClientAPI:
    """
    Initialize persistent ChromaDB client inside ./chroma_db.
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Get or create the legal documents collection in ChromaDB.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Intelligent Legal Document RAG Collection"}
    )

# -----------------------------------------------------------------------------
# 5. PDF Processing & Text Cleaning
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalize whitespace, handle linebreaks, and clean up raw PDF text extraction artifacts.
    """
    if not text:
        return ""
    # Normalize carriage returns and newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse consecutive horizontal whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse excessive vertical whitespace (more than 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(uploaded_file) -> Dict[str, Any]:
    """
    Extracts text page by page from an uploaded PDF stream using pypdf.
    Returns metadata including total pages, page texts, and status.
    """
    try:
        # Check if file has 0 bytes
        uploaded_file.seek(0, os.SEEK_END)
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)
        
        if file_size == 0:
            return {
                "success": False,
                "error": "The uploaded PDF file is empty (0 bytes). Please upload a valid legal document.",
                "pages": []
            }

        reader = PdfReader(uploaded_file)
        num_pages = len(reader.pages)

        if num_pages == 0:
            return {
                "success": False,
                "error": "The uploaded PDF has 0 pages or is corrupted.",
                "pages": []
            }

        pages_data = []
        total_text_length = 0

        for idx, page in enumerate(reader.pages):
            page_number = idx + 1
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)
            pages_data.append({
                "page": page_number,
                "text": cleaned
            })
            total_text_length += len(cleaned)

        # Detect scanned PDFs (0 extractable text across all pages)
        if total_text_length == 0:
            return {
                "success": False,
                "error": "No selectable text could be extracted from this PDF. It appears to be a scanned image or image-only document. Please upload a PDF containing selectable text.",
                "pages": []
            }

        return {
            "success": True,
            "error": None,
            "total_pages": num_pages,
            "pages": pages_data
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read PDF document: {str(e)}",
            "pages": []
        }

# -----------------------------------------------------------------------------
# 6. Page-Preserving Text Chunking
# -----------------------------------------------------------------------------
def chunk_document(
    pages_data: List[Dict[str, Any]],
    source_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[DocumentChunk]:
    """
    Splits document page-by-page into overlapping character chunks.
    Crucially preserves exact page number metadata for every chunk.
    
    Example:
      Page 1 -> Chunk 1
      Page 1 -> Chunk 2
      Page 2 -> Chunk 3
      Page 3 -> Chunk 4
    """
    chunks: List[DocumentChunk] = []
    total_chunk_counter = 0

    for page_item in pages_data:
        page_num = page_item["page"]
        text = page_item["text"]
        if not text:
            continue

        text_len = len(text)
        # If the page text fits comfortably in one chunk
        if text_len <= chunk_size:
            total_chunk_counter += 1
            chunk_id = f"{source_name}_p{page_num}_c{total_chunk_counter}"
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=text,
                page=page_num,
                chunk_number=total_chunk_counter,
                source=source_name
            ))
        else:
            # Sliding window with overlap
            start = 0
            stride = chunk_size - overlap
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_str = text[start:end].strip()

                # Filter out tiny residual fragments (< 30 characters)
                if len(chunk_str) >= 30:
                    total_chunk_counter += 1
                    chunk_id = f"{source_name}_p{page_num}_c{total_chunk_counter}"
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        text=chunk_str,
                        page=page_num,
                        chunk_number=total_chunk_counter,
                        source=source_name
                    ))

                if end == text_len:
                    break
                start += stride

    return chunks

# -----------------------------------------------------------------------------
# 7. Embedding Generation & ChromaDB Storage
# -----------------------------------------------------------------------------
def index_document_in_chroma(
    collection: chromadb.Collection,
    chunks: List[DocumentChunk],
    embedding_model: SentenceTransformer
) -> int:
    """
    Generates embeddings for each chunk using SentenceTransformers and
    stores them into ChromaDB with comprehensive metadata (source, page, chunk).
    """
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    ids = [c.id for c in chunks]
    metadatas = [
        {
            "source": c.source,
            "page": int(c.page),
            "chunk": int(c.chunk_number)
        }
        for c in chunks
    ]

    # Generate embeddings batch
    embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()

    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    return len(chunks)

# -----------------------------------------------------------------------------
# 8. Similarity Search & Context Retrieval
# -----------------------------------------------------------------------------
def retrieve_relevant_chunks(
    collection: chromadb.Collection,
    question: str,
    embedding_model: SentenceTransformer,
    top_k: int = 5,
    source_name: Optional[str] = None
) -> List[RetrievedSource]:
    """
    Encodes the user question and performs cosine similarity search in ChromaDB.
    Returns the top-k most relevant legal chunks.
    """
    # Generate question embedding
    q_embedding = embedding_model.encode(question).tolist()

    # Filter by source document if specified
    where_filter = {"source": source_name} if source_name else None

    # Query ChromaDB collection
    query_results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        where=where_filter
    )

    retrieved: List[RetrievedSource] = []
    if query_results and "documents" in query_results and query_results["documents"]:
        docs = query_results["documents"][0]
        metas = query_results["metadatas"][0] if "metadatas" in query_results else [{}] * len(docs)
        distances = query_results["distances"][0] if "distances" in query_results else [0.0] * len(docs)
        ids = query_results["ids"][0] if "ids" in query_results else [""] * len(docs)

        for doc_text, meta, dist, cid in zip(docs, metas, distances, ids):
            retrieved.append(RetrievedSource(
                id=cid,
                text=doc_text,
                page=meta.get("page", 1),
                chunk=meta.get("chunk", 1),
                source=meta.get("source", "Document"),
                distance=dist
            ))

    return retrieved

# -----------------------------------------------------------------------------
# 9. LLM Answer Generation via HTTP Requests
# -----------------------------------------------------------------------------
def generate_answer(
    question: str,
    context_chunks: List[RetrievedSource],
    api_key: str,
    api_url: str = "https://api.groq.com/openai/v1/chat/completions",
    model_name: str = "llama-3.1-8b-instant"
) -> Dict[str, Any]:
    """
    Sends retrieved legal context and user question to an LLM via the requests library.
    Instructs the model with strict grounding, legal explanation in plain terms, and page citations.
    """
    if not api_key or api_key.strip() == "" or api_key.strip() == "your_api_key_here":
        return {
            "success": False,
            "error_type": "MISSING_API_KEY",
            "error": "LLM API Key is missing. Please provide your API key in the sidebar or in the .env file."
        }

    # Format the retrieved chunks with page markers
    formatted_context = ""
    for idx, c in enumerate(context_chunks, 1):
        formatted_context += (
            f"--- [EXCERPT {idx}] (Document: '{c.source}', Page: {c.page}, Chunk: #{c.chunk}) ---\n"
            f"{c.text}\n\n"
        )

    # Prompt Engineering with Strict Legal Guardrails
    system_instruction = (
        "You are an expert AI Legal Document Assistant built for educational document analysis. "
        "Your task is to answer user questions based STRICTLY and ONLY on the provided legal excerpts.\n\n"
        "STRICT COMPLIANCE RULES:\n"
        "1. Answer ONLY using information directly stated in the provided excerpts.\n"
        "2. Do NOT invent, assume, extrapolate, or introduce outside knowledge or facts.\n"
        "3. If the answer cannot be determined from the provided excerpts, reply EXACTLY with:\n"
        "   'The answer was not found in the uploaded document.'\n"
        "4. Explain any legal terminology or jargon in simple, beginner-friendly terms.\n"
        "5. Always reference specific page numbers when citing clauses or facts (e.g., 'According to Page 2...').\n"
        "6. Do NOT provide personalized legal advice, and do NOT state or imply that you are acting as an attorney.\n"
        "7. Format your answer with clear, structured bullet points or paragraphs for maximum readability."
    )

    user_prompt = (
        f"RETRIEVED LEGAL CONTEXT:\n"
        f"{formatted_context}\n"
        f"USER QUESTION:\n"
        f"{question}\n\n"
        f"Please provide an accurate, clear answer adhering strictly to the compliance rules."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,  # Low temperature for factual, deterministic legal extraction
        "max_tokens": 1200
    }

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=45
        )

        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                answer = data["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "answer": answer.strip(),
                    "model": model_name
                }
            else:
                return {
                    "success": False,
                    "error_type": "EMPTY_RESPONSE",
                    "error": "The LLM API returned an empty choices list in its response."
                }
        elif response.status_code == 401:
            return {
                "success": False,
                "error_type": "AUTH_ERROR",
                "error": "Authentication failed (401 Unauthorized). Please check that your LLM API Key is valid."
            }
        elif response.status_code == 429:
            return {
                "success": False,
                "error_type": "RATE_LIMIT",
                "error": "Rate limit exceeded (429 Too Many Requests). Please wait a few moments before asking another question."
            }
        else:
            return {
                "success": False,
                "error_type": "API_ERROR",
                "error": f"API request failed with HTTP status {response.status_code}: {response.text[:300]}"
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_type": "TIMEOUT",
            "error": "The LLM API request timed out (45s). Please check your internet connection or try again."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error_type": "CONNECTION_ERROR",
            "error": "Failed to connect to the LLM API endpoint. Please check your internet connection and the API URL."
        }
    except Exception as e:
        return {
            "success": False,
            "error_type": "UNKNOWN_ERROR",
            "error": f"An unexpected error occurred during API communication: {str(e)}"
        }

# -----------------------------------------------------------------------------
# 10. Session State Initialization
# -----------------------------------------------------------------------------
if "processed_doc" not in st.session_state:
    st.session_state["processed_doc"] = None  # Holds active doc metadata
if "retrieved_sources" not in st.session_state:
    st.session_state["retrieved_sources"] = []
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = None
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""
if "selected_example_question" not in st.session_state:
    st.session_state["selected_example_question"] = ""

# -----------------------------------------------------------------------------
# 11. Sidebar: Controls, Example Questions, & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚖️ Project Navigation")
    st.markdown('<span class="rag-badge">RAG Architecture Active</span>', unsafe_allow_html=True)
    
    # Clear / New Document Button
    if st.button("🔄 Clear / New Document", use_container_width=True, help="Reset session and upload another document"):
        st.session_state["processed_doc"] = None
        st.session_state["retrieved_sources"] = []
        st.session_state["last_answer"] = None
        st.session_state["last_question"] = ""
        st.session_state["selected_example_question"] = ""
        st.rerun()

    st.markdown("---")
    
    # Example Questions Section
    st.markdown("### 💡 Example Questions")
    st.caption("Click any question below to populate the input box:")
    
    example_questions = [
        "What is this agreement about?",
        "What are the responsibilities of the parties?",
        "What are the termination conditions?",
        "What happens if the agreement is breached?",
        "What is the duration of this agreement?",
        "What are the important obligations?",
        "Are there any penalties mentioned in the document?"
    ]
    
    for q in example_questions:
        if st.button(f"📌 {q}", key=f"btn_ex_{hash(q)}", use_container_width=True):
            st.session_state["selected_example_question"] = q
            st.rerun()

    st.markdown("---")

    # LLM & API Configuration Section
    st.markdown("### ⚙️ LLM & API Settings")
    
    # Read from .env as defaults
    env_api_key = os.getenv("LLM_API_KEY", "")
    env_api_url = os.getenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    env_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    
    user_api_key = st.text_input(
        "API Key (Groq or OpenAI)",
        value=env_api_key if env_api_key != "your_api_key_here" else "",
        type="password",
        help="Reads LLM_API_KEY from .env by default. Supports Groq (free) or OpenAI.",
        placeholder="Enter your API key..."
    )
    
    user_api_url = st.text_input(
        "API URL",
        value=env_api_url,
        help="Endpoint for completions using requests. Default: Groq OpenAI-compatible endpoint."
    )
    
    user_model = st.text_input(
        "Model Name",
        value=env_model,
        help="e.g. llama-3.1-8b-instant, gpt-4o-mini, mixtral-8x7b-32768"
    )

    # API Status Indicator
    if user_api_key and user_api_key != "your_api_key_here":
        st.success("🟢 API Key Configured")
    else:
        st.warning("🟡 No API Key set. You can still inspect PDF chunks & vector search.")

    st.markdown("---")
    st.markdown("### 🎓 BTech Viva Reference")
    with st.expander("ℹ️ Technology Stack Details"):
        st.markdown("""
        * **UI**: Streamlit v1.40.1
        * **Extraction**: PyPDF v5.1.0
        * **Embeddings**: `all-MiniLM-L6-v2` (384-d)
        * **Vector DB**: ChromaDB v0.5.23
        * **HTTP Client**: Requests v2.32.3
        * **Chunk Size**: 1000 chars (200 overlap)
        """)

# -----------------------------------------------------------------------------
# 12. Main Dashboard Interface
# -----------------------------------------------------------------------------
# Title and Subtitle
st.markdown('<div class="legal-title">⚖️ Intelligent Legal Document Answering</div>', unsafe_allow_html=True)
st.markdown('<div class="legal-subtitle">Upload a legal document and ask questions about its contents.</div>', unsafe_allow_html=True)

# Important Legal Safety Notice
st.markdown("""
<div class="legal-disclaimer">
    ⚠️ <strong>Important Legal Safety Notice:</strong> This application is an AI-based document analysis tool for educational and informational purposes only. It does not provide professional legal advice. Always consult a qualified legal professional for legal decisions.
</div>
""", unsafe_allow_html=True)

# Load Embedding Model and ChromaDB Client
try:
    with st.spinner("Initializing Sentence Transformers & ChromaDB..."):
        embedding_model = load_embedding_model()
        chroma_client = get_chroma_client()
        collection = get_or_create_collection(chroma_client)
except Exception as e:
    st.error(f"Initialization Error: Could not initialize embedding model or ChromaDB: {e}")
    st.stop()

# Layout: Split into Document Upload & QA Panels
col1, col2 = st.columns([1, 1], gap="large")

# =============================================================================
# COLUMN 1: PDF Upload & Document Processing Status
# =============================================================================
with col1:
    st.markdown("### 📄 1. Document Upload & Processing")
    
    uploaded_pdf = st.file_uploader(
        "Upload Legal PDF Document",
        type=["pdf"],
        help="Upload agreements, contracts, NDAs, or terms of service in PDF format."
    )

    if uploaded_pdf is not None:
        doc_filename = uploaded_pdf.name
        
        # Check if this document is already processed in session state
        already_processed = (
            st.session_state["processed_doc"] is not None and 
            st.session_state["processed_doc"].get("name") == doc_filename
        )

        if not already_processed:
            # Process the PDF document
            with st.status(f"Processing '{doc_filename}'...", expanded=True) as status:
                st.write("📖 Reading PDF pages with PyPDF...")
                extraction_result = extract_text_from_pdf(uploaded_pdf)
                
                if not extraction_result["success"]:
                    status.update(label="❌ PDF Processing Failed", state="error", expanded=True)
                    st.error(f"Error: {extraction_result['error']}")
                else:
                    pages_data = extraction_result["pages"]
                    total_pages = extraction_result["total_pages"]
                    st.write(f"✅ Extracted text from {total_pages} page(s).")
                    
                    st.write(f"✂️ Splitting into chunks (~{CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)...")
                    chunks = chunk_document(pages_data, doc_filename, CHUNK_SIZE, CHUNK_OVERLAP)
                    st.write(f"✅ Created {len(chunks)} text chunk(s) preserving page numbers.")
                    
                    st.write(f"🧠 Generating SentenceTransformer embeddings (`{EMBEDDING_MODEL_NAME}`)...")
                    indexed_count = index_document_in_chroma(collection, chunks, embedding_model)
                    st.write(f"💾 Stored {indexed_count} chunks in ChromaDB vector database.")
                    
                    # Save to session state
                    st.session_state["processed_doc"] = {
                        "name": doc_filename,
                        "pages": total_pages,
                        "chunks_count": len(chunks),
                        "chunks": [c.model_dump() for c in chunks]
                    }
                    status.update(label=f"✅ '{doc_filename}' Processed Successfully!", state="complete", expanded=False)
                    st.success(f"Successfully processed and indexed '{doc_filename}'!")

        # Display Document Metadata Cards
        if st.session_state["processed_doc"]:
            pdoc = st.session_state["processed_doc"]
            st.markdown("#### 📊 Document Summary")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{pdoc['pages']}</div>
                    <div class="stat-label">Total Pages</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{pdoc['chunks_count']}</div>
                    <div class="stat-label">Text Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #10B981;">Ready</div>
                    <div class="stat-label">Vector DB Status</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption(f"📁 **Active Document:** `{pdoc['name']}`")
            
            # Expandable Chunk Inspector for Viva
            with st.expander("🔍 Inspect Generated Chunks & Page Metadata"):
                st.write(f"Displaying first {min(5, len(pdoc['chunks']))} chunks of {len(pdoc['chunks'])} total:")
                for c in pdoc['chunks'][:5]:
                    st.markdown(f"**Chunk #{c['chunk_number']} (Page {c['page']})** - *ID: `{c['id']}`*")
                    st.text(c['text'][:300] + ("..." if len(c['text']) > 300 else ""))
                    st.divider()

    else:
        # Prompt user to upload
        st.info("👈 Please upload a legal PDF document above to get started.")
        st.markdown("""
        **What happens when you upload?**
        1. **Text Extraction**: PyPDF reads each page separately.
        2. **Page-Preserving Chunking**: Text is split into 1000-character chunks while recording its exact page number.
        3. **Vector Embeddings**: `all-MiniLM-L6-v2` converts chunks into semantic vectors.
        4. **Vector Storage**: Embeddings and metadata are indexed in local ChromaDB for fast similarity retrieval.
        """)

# =============================================================================
# COLUMN 2: Question Answering & Sources Display
# =============================================================================
with col2:
    st.markdown("### 💬 2. Ask Questions About the Document")
    
    # Prefill question if user clicked an example question in the sidebar
    initial_question = ""
    if st.session_state["selected_example_question"]:
        initial_question = st.session_state["selected_example_question"]
        st.session_state["selected_example_question"] = ""  # Reset after applying
    
    user_question = st.text_input(
        "Enter your question:",
        value=initial_question,
        placeholder="e.g., What are the termination conditions? What is the duration of this agreement?",
        key="question_input_widget"
    )

    ask_button = st.button("🚀 Ask Question", use_container_width=True, type="primary")

    if ask_button:
        # 1. Validation: Document must be uploaded
        if not st.session_state["processed_doc"]:
            st.warning("⚠️ Please upload a legal PDF document first before asking questions.")
        # 2. Validation: Question must not be empty
        elif not user_question or user_question.strip() == "":
            st.warning("⚠️ Please enter a question in the box above.")
        else:
            active_doc_name = st.session_state["processed_doc"]["name"]
            
            with st.spinner("🔍 Performing similarity search in ChromaDB..."):
                retrieved_chunks = retrieve_relevant_chunks(
                    collection=collection,
                    question=user_question.strip(),
                    embedding_model=embedding_model,
                    top_k=5,
                    source_name=active_doc_name
                )
                st.session_state["retrieved_sources"] = retrieved_chunks
                st.session_state["last_question"] = user_question.strip()

            if not retrieved_chunks:
                st.error("No relevant document sections found for this query.")
            else:
                # Call LLM via Requests
                with st.spinner("🤖 Generating legal answer using LLM..."):
                    llm_result = generate_answer(
                        question=user_question.strip(),
                        context_chunks=retrieved_chunks,
                        api_key=user_api_key,
                        api_url=user_api_url,
                        model_name=user_model
                    )

                if llm_result["success"]:
                    st.session_state["last_answer"] = llm_result["answer"]
                else:
                    # If API key is missing or failed, provide clear message
                    if llm_result.get("error_type") == "MISSING_API_KEY":
                        st.warning("🔑 **API Key Needed for Live Answer Generation**")
                        st.info(
                            "To get live AI answers, enter your API key in the sidebar or in `.env`.\n\n"
                            "👉 **Free Option**: You can get a free, instant Groq API key at [console.groq.com](https://console.groq.com/keys).\n\n"
                            "Meanwhile, **semantic search has succeeded!** You can inspect the retrieved relevant sources below."
                        )
                        st.session_state["last_answer"] = (
                            "*(API Key missing. The RAG retrieval pipeline successfully found the relevant document chunks below. "
                            "Add your API key to generate an automated synthesis.)*"
                        )
                    else:
                        st.error(f"❌ Error generating answer: {llm_result.get('error')}")
                        st.session_state["last_answer"] = None

    # Display Answer Section
    if st.session_state["last_answer"]:
        st.markdown("---")
        st.markdown("### 📝 Answer")
        st.info(f"**Q: {st.session_state['last_question']}**")
        st.markdown(st.session_state["last_answer"])

    # Display Retrieved Sources Section
    if st.session_state["retrieved_sources"]:
        st.markdown("---")
        st.markdown("### 📚 Sources")
        st.caption(f"Retrieved {len(st.session_state['retrieved_sources'])} most relevant chunk(s) from ChromaDB:")

        for i, src in enumerate(st.session_state["retrieved_sources"], 1):
            with st.expander(f"📍 Source {i} — Page {src.page} (Chunk #{src.chunk})"):
                st.markdown(f"**Document**: `{src.source}` | **Page**: `{src.page}` | **Chunk**: `#{src.chunk}` | **Similarity Distance**: `{src.distance:.4f}`")
                st.markdown(f"```text\n{src.text}\n```")

# -----------------------------------------------------------------------------
# 13. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "⚖️ <strong>Intelligent Legal Document Answering</strong> | BTech Capstone Project | Built with Streamlit, PyPDF, SentenceTransformers & ChromaDB"
    "</div>",
    unsafe_allow_html=True
)
