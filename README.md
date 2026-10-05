# ⚖️ Intelligent Legal Document Answering

> An AI-powered Retrieval-Augmented Generation (RAG) system for analyzing and querying legal agreements, contracts, and documents with exact page-level source attribution.

---

## 📌 1. Project Title
**Intelligent Legal Document Answering**  
*College BTech Capstone / AI Project*

---

## 🎯 2. Project Objective
Legal contracts and agreements are notoriously dense, verbose, and difficult for non-specialists to navigate. The primary objective of this project is to build an intelligent, reliable question-answering system that allows users to:
1. Upload complex legal documents in PDF format.
2. Automatically parse, clean, and segment legal clauses while strictly preserving page numbers.
3. Convert legal text into semantic vector embeddings.
4. Perform similarity search to retrieve the exact clauses answering a user's question.
5. Generate an objective, easily understandable answer synthesized strictly from the retrieved document text.
6. Display transparent source citations with exact page numbers and chunk IDs so the user can verify every single claim.

---

## ✨ 3. Features
* **📄 Multi-Page PDF Ingestion**: Page-by-page text extraction using `pypdf`.
* **✂️ Page-Preserving Text Chunking**: Splits text into 1,000-character overlapping chunks (~200 characters overlap) while tagging each chunk with its 1-indexed page number.
* **🧠 High-Performance Local Embeddings**: Uses `sentence-transformers` with the `all-MiniLM-L6-v2` model (384-dimensional vector space).
* **💾 Persistent Vector Database**: Uses `ChromaDB` for high-speed nearest-neighbor similarity search and metadata storage.
* **🤖 Grounded LLM Generation**: Uses the `requests` library to communicate with OpenAI-compatible LLM endpoints (such as Groq's high-speed Llama 3.1 or OpenAI GPT-4o-mini).
* **🛡️ Hallucination Safeguards**: Prompts strictly restrict the model from inventing information or offering personalized legal advice. If a clause is absent, it reports: *"The answer was not found in the uploaded document."*
* **📚 Expandable Source Inspector**: Interactive Streamlit expanders display the exact retrieved chunks, similarity distances, and source pages.
* **💡 One-Click Example Questions**: Pre-loaded legal queries in the sidebar for quick testing during demonstrations and vivas.
* **⚠️ Legal Safety Notice**: Built-in educational disclaimer ensuring compliance with ethical AI standards.

---

## 🛠️ 4. Technologies Used
The project is built with the following core Python libraries and exact versions:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| **`streamlit`** | `1.40.1` | Modern, responsive web user interface |
| **`python-dotenv`** | `1.0.1` | Secure management of `.env` API keys and configurations |
| **`requests`** | `2.32.3` | Clean HTTP calls to LLM API endpoints without heavy SDK bloat |
| **`pypdf`** | `5.1.0` | Reliable PDF parsing and per-page text extraction |
| **`chromadb`** | `0.5.23` | Local vector database for embedding indexing and similarity search |
| **`sentence-transformers`** | `3.3.1` | State-of-the-art dense semantic embedding generation |
| **`pydantic`** | `2.10.3` | Data validation and structured schemas for document chunks and responses |

---

## 🏛️ 5. RAG Architecture

```text
               +-----------------------------+
               |       Legal PDF File        |
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               |     PDF Text Extraction     |  <-- pypdf (PdfReader)
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               |  Text Cleaning & Filtering  |  <-- Whitespace normalization
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               |    Page-Preserving Chunking |  <-- 1000 chars, 200 overlap
               +-----------------------------+      (Tags page: N, chunk: M)
                              |
                              v
               +-----------------------------+
               | SentenceTransformer Embed   |  <-- all-MiniLM-L6-v2 (384-d)
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               |       ChromaDB Storage      |  <-- ./chroma_db/
               +-----------------------------+
                              |
     User Question            |
           |                  |
           v                  |
   [Question Embedding]       |
           |                  |
           v                  |
   [Cosine Similarity Search] <
           |
           v
   +-----------------------------+
   |   Top 5 Relevant Chunks     |  <-- Filtered by document & ranked by distance
   +-----------------------------+
                 |
                 v
   +-----------------------------+
   |  LLM Prompt Construction    |  <-- Strict grounding & page citations
   +-----------------------------+
                 |
                 v
   +-----------------------------+
   |  LLM Generation via requests|  <-- Groq / OpenAI endpoint
   +-----------------------------+
                 |
                 v
   +-----------------------------+
   | Final Answer + Cited Sources|  <-- Streamlit UI Display
   +-----------------------------+
```

---

## 🔍 6. How PDF Processing Works
1. When a PDF is uploaded, the stream is validated for non-zero file size.
2. `pypdf.PdfReader` reads the file page-by-page.
3. For each page `i` (1 to `N`), `page.extract_text()` extracts the raw textual content.
4. If total extractable text across all pages is zero, the system raises a clear notification explaining that the PDF is likely a scanned image requiring OCR.
5. The extracted text is cleaned by stripping excessive consecutive newlines, carriage returns, and multi-space gaps.
6. A sliding window splits each page into chunks of **1,000 characters** with an overlap of **200 characters**, ensuring context around clause boundaries is never lost.
7. Crucially, each chunk retains metadata:
   * `source`: Document filename
   * `page`: Source page number
   * `chunk`: Sequential chunk index

---

## 🧠 7. How Embeddings Work
* Text cannot be queried mathematically until it is represented as a dense numerical vector.
* We utilize `sentence-transformers` with the model `all-MiniLM-L6-v2`.
* It maps sentences and paragraphs to a **384-dimensional dense vector space**.
* The model is cached in memory using Streamlit's `@st.cache_resource` decorator, so it loads only once upon startup.
* When querying, the user's natural language question is converted into a 384-dimensional vector and compared against chunk vectors using cosine distance.

---

## 🗄️ 8. How ChromaDB Works
* `ChromaDB` acts as the persistent vector database stored locally in `./chroma_db`.
* Documents are stored in a collection named `legal_documents`.
* For each chunk, ChromaDB stores:
  * `id`: e.g., `Contract_p1_c1`
  * `document`: The raw chunk text
  * `embedding`: The 384-float vector from SentenceTransformers
  * `metadata`: `{"source": "NDA.pdf", "page": 1, "chunk": 1}`
* Similarity search executes a nearest-neighbor query (`collection.query(...)`) to return the **top 5** most semantically relevant chunks.

---

## 🤖 9. How Question Answering Works
1. The retrieved chunks are formatted into a context block with page headers:
   `--- [EXCERPT 1] (Document: 'NDA.pdf', Page: 2, Chunk: #3) ---`
2. A system prompt establishes strict boundaries:
   * Must answer exclusively from the retrieved excerpts.
   * If the answer is not present, respond: *"The answer was not found in the uploaded document."*
   * Explain legal jargon in plain terms.
   * Cite exact page numbers.
   * Never offer personalized legal advice.
3. The prompt is sent to the LLM via standard HTTP POST using the Python `requests` library:
   ```python
   response = requests.post(api_url, headers=headers, json=payload, timeout=45)
   ```
4. The response text is extracted and presented alongside interactive source expanders.

---

## 📥 10. Installation Instructions

### Prerequisites
* Python 3.10 or 3.11 installed.
* VS Code or your preferred IDE.

### Step-by-Step Setup

1. **Open the project directory in VS Code or Terminal**:
   ```powershell
   cd Intelligent-Legal-Document-Answering
   ```

2. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   * **Windows PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Windows Command Prompt (cmd)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   * **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required packages**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ⚙️ 11. `.env` Configuration
Copy the template or edit `.env` in the root folder:

```ini
# ==============================================================================
# Intelligent Legal Document Answering - Configuration
# ==============================================================================

# Your API Key (Groq or OpenAI)
LLM_API_KEY=your_api_key_here

# API Endpoint (Groq provides free, instant keys at console.groq.com)
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions

# Model Name
LLM_MODEL=llama-3.1-8b-instant
```

> **Tip for Students / Viva**: You can get a completely free Groq API key in 30 seconds at [console.groq.com/keys](https://console.groq.com/keys). Alternatively, any OpenAI or OpenAI-compatible endpoint works out-of-the-box. You can also paste your key directly into the Streamlit sidebar at runtime.

---

## 🚀 12. How to Run the Project

Ensure your virtual environment is activated, then run:

```powershell
python -m streamlit run app.py
```

The application will launch in your browser automatically at:
```text
http://localhost:8501
```

---

## 💡 13. Example Questions
Test the system with typical legal contract inquiries:
1. *What is this agreement about?*
2. *What are the responsibilities of the parties?*
3. *What are the termination conditions?*
4. *What happens if the agreement is breached?*
5. *What is the duration of this agreement?*
6. *What are the important obligations?*
7. *Are there any penalties mentioned in the document?*

---

## ⚠️ 14. Limitations
* **Scanned Images**: Text extraction requires text-based PDFs. Pure image scans without an OCR layer will not yield extractable text.
* **Tables and Layout Complexity**: Multi-column text or complex borderless tables may have non-linear reading order in raw text streams.
* **Language Support**: Default embeddings (`all-MiniLM-L6-v2`) are optimized primarily for English legal documents.

---

## 🔮 15. Future Enhancements
* **Tesseract OCR Integration**: Automatic OCR fallback for scanned contracts.
* **Hybrid Search (BM25 + Dense Vectors)**: Combining keyword matching for specific legal clause numbers with dense semantic search.
* **Multi-Document Comparison**: Querying across two versions of a contract to highlight amendments and risk deltas.
* **Clause Risk Rating**: Flagging high-liability or non-standard indemnity clauses using specialized classifiers.

---

## ⚖️ Important Legal Safety Notice
> **"This application is an AI-based document analysis tool for educational and informational purposes only. It does not provide professional legal advice. Always consult a qualified legal professional for legal decisions."**

---

## 🎓 Viva Questions & Answers (For BTech Students)

**Q1: What is RAG and why use it instead of fine-tuning?**  
*Answer:* Retrieval-Augmented Generation (RAG) retrieves private, up-to-date document chunks and injects them into the LLM's context window at inference time. Fine-tuning teaches models new styles or domains but cannot guarantee strict factual retrieval or exact page-level citations, and is prone to hallucinating facts.

**Q2: Why use chunk overlap?**  
*Answer:* If a key sentence or legal clause spans across the boundary of Chunk 1 and Chunk 2, a hard split might cut the sentence in half, destroying its semantic meaning. A 200-character overlap guarantees that every boundary sentence is preserved intact in at least one chunk.

**Q3: How does ChromaDB perform similarity search?**  
*Answer:* ChromaDB computes the cosine distance (or dot product) between the query vector and all stored document chunk vectors using an HNSW (Hierarchical Navigable Small World) index, returning the closest matches in milliseconds.
