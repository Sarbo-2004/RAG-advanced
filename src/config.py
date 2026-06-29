import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PDF_DIR = "data/pdfs"
VECTORSTORE_DIR = "vectorstore/faiss_index"
DOCLING_VECTORSTORE_DIR = "vectorstore/faiss_docling_index"

EMBEDDING_MODEL = r"D:\OneDrive - Coforge Limited\Desktop\RAG-advanced\hugging_face_model\all-MiniLM-L6-v2"
RERANKER_MODEL = r"D:\OneDrive - Coforge Limited\Desktop\RAG-advanced\hugging_face_model\ms-marco-MiniLM-L12-v2"
LLM_MODEL = "gemini-flash-lite-latest"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


RETRIEVER_K = 8
RERANK_TOP_K = 3

MAX_HISTORY_MESSAGES = 6
ANSWER_TEMPERATURE = 0.2

