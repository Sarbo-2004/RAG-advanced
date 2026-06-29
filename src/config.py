"""Central configuration and logging for the FAO RAG chatbot.

All tunables live here. Model identifiers default to Hugging Face Hub names so
the app works on any machine; override any value with an environment variable
(e.g. EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL) to point at a local copy.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# Secrets
# --------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# Paths
# --------------------------------------------------
PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
DOCLING_VECTORSTORE_DIR = os.getenv("DOCLING_VECTORSTORE_DIR", "vectorstore/faiss_docling_index")

# Optional local Docling model cache (only needed when rebuilding the index).
# If unset, Docling downloads its models automatically on first use.
DOCLING_MODELS_PATH = os.getenv("DOCLING_MODELS_PATH") or None


# --------------------------------------------------
# Models
# --------------------------------------------------
# Hugging Face Hub identifiers. Resolve to a local directory if one is passed,
# otherwise downloaded from the Hub on first use.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-lite-latest")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


# --------------------------------------------------
# Chunking (used by the offline index builder)
# --------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


# --------------------------------------------------
# Retrieval / reranking
# --------------------------------------------------
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))        # candidates pulled from FAISS
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))      # kept after cross-encoder rerank


# --------------------------------------------------
# Conversation / generation
# --------------------------------------------------
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.2"))


# --------------------------------------------------
# Prompts
# --------------------------------------------------
QA_PROMPT_PATH = os.getenv("QA_PROMPT_PATH", "prompts/qa_prompt.txt")
CONTEXTUALIZE_PROMPT_PATH = os.getenv("CONTEXTUALIZE_PROMPT_PATH", "prompts/contextualize_prompt.txt")


# --------------------------------------------------
# Logging
# --------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_logging_configured = False


def configure_logging() -> None:
    """Configure root logging once, idempotently."""
    global _logging_configured

    if _logging_configured:
        return

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    configure_logging()
    return logging.getLogger(name)


def validate_runtime_config() -> None:
    """Fail fast with a clear message if required runtime config is missing.

    Call this from the application entrypoint (not at import time) so that
    offline tooling and unit tests can import this module without a key.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to a .env file or your "
            "environment before starting the app. See .env.example."
        )

    if not os.path.exists(DOCLING_VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at '{DOCLING_VECTORSTORE_DIR}'. "
            "Run `python build_docling_faiss_index.py` to build it."
        )
