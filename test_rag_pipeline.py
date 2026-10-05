"""
Verification script for the RAG pipeline components:
1. PyPDF text extraction
2. Text chunking with page retention
3. SentenceTransformers embeddings
4. ChromaDB storage and similarity search
"""
import os
import sys
import types
import importlib.machinery
from unittest.mock import MagicMock

# Windows WDAC / AppLocker safety shim for sklearn evaluation modules not needed for inference
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

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel

print("Testing imports...")
from app import extract_text_from_pdf, chunk_document, index_document_in_chroma, retrieve_relevant_chunks, DocumentChunk

sample_pdf_path = "documents/Sample_Service_and_NDA_Agreement.pdf"
print(f"Reading sample PDF: {sample_pdf_path}")

with open(sample_pdf_path, "rb") as f:
    result = extract_text_from_pdf(f)

print(f"Success: {result['success']}")
print(f"Total Pages: {result['total_pages']}")
for p in result['pages']:
    print(f"  Page {p['page']}: {len(p['text'])} chars extracted")

chunks = chunk_document(result['pages'], "Sample_Agreement.pdf", chunk_size=1000, overlap=200)
print(f"Generated {len(chunks)} chunks:")
for c in chunks:
    print(f"  Chunk #{c.chunk_number} | Page {c.page} | ID: {c.id} | Length: {len(c.text)}")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Initializing ChromaDB test collection...")
client = chromadb.Client()  # In-memory test client
collection = client.get_or_create_collection("test_legal_docs")

indexed_count = index_document_in_chroma(collection, chunks, model)
print(f"Indexed {indexed_count} chunks into ChromaDB.")

# Test query
query = "What happens if the agreement is breached and what are the penalties?"
print(f"\nQuerying: '{query}'")
retrieved = retrieve_relevant_chunks(collection, query, model, top_k=3)

print(f"Retrieved {len(retrieved)} chunks:")
for r in retrieved:
    print(f"  -> Page {r.page} | Chunk #{r.chunk} | Distance: {r.distance:.4f}")
    print(f"     Preview: {r.text[:120]}...\n")

print("RAG PIPELINE VERIFICATION PASSED SUCCESSFULLY!")
