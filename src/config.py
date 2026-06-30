"""Central configuration for the FAO RAG chatbot."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# Base Directory
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Secrets
# --------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# Paths
# --------------------------------------------------
PDF_DIR = os.getenv(
    "PDF_DIR",
    str(BASE_DIR / "data" / "pdfs")
)

DOCLING_VECTORSTORE_DIR = os.getenv(
    "DOCLING_VECTORSTORE_DIR",
    str(BASE_DIR / "vectorstore" / "faiss_docling_index")
)

DOCLING_MODELS_PATH = os.getenv(
    "DOCLING_MODELS_PATH",
    str(BASE_DIR / "docling_models")
)


# --------------------------------------------------
# Local Models
# --------------------------------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    str(BASE_DIR / "hugging_face_model" / "all-MiniLM-L6-v2")
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    str(BASE_DIR / "hugging_face_model" / "ms-marco-MiniLM-L12-v2")
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-1.5-flash"
)

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


# --------------------------------------------------
# Chunking
# --------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


# --------------------------------------------------
# Retrieval / Reranking
# --------------------------------------------------
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))


# --------------------------------------------------
# Conversation / Generation
# --------------------------------------------------
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.2"))


# --------------------------------------------------
# Prompts
# --------------------------------------------------
QA_PROMPT_PATH = os.getenv(
    "QA_PROMPT_PATH",
    str(BASE_DIR / "prompts" / "qa_prompt.txt")
)

CONTEXTUALIZE_PROMPT_PATH = os.getenv(
    "CONTEXTUALIZE_PROMPT_PATH",
    str(BASE_DIR / "prompts" / "contextualize_prompt.txt")
)


# --------------------------------------------------
# Runtime Validation
# --------------------------------------------------
def validate_runtime_config() -> None:
    """Validate required runtime configuration."""

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file."
        )

    if not os.path.exists(PDF_DIR):
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIR}"
        )

    if not os.path.exists(DOCLING_VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at: {DOCLING_VECTORSTORE_DIR}. "
            "Run build_docling_faiss_index.py first."
        )

    if not os.path.exists(EMBEDDING_MODEL):
        raise FileNotFoundError(
            f"Embedding model not found at: {EMBEDDING_MODEL}"
        )

    if not os.path.exists(RERANKER_MODEL):
        raise FileNotFoundError(
            f"Reranker model not found at: {RERANKER_MODEL}"
        )

    if not os.path.exists(DOCLING_MODELS_PATH):
        raise FileNotFoundError(
            f"Docling models not found at: {DOCLING_MODELS_PATH}"
        )