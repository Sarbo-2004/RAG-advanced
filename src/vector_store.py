"""Embedding model and FAISS index loading for the Docling pipeline."""

import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    DOCLING_VECTORSTORE_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    get_logger,
)

logger = get_logger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local/Hub Hugging Face embedding model.

    Using a local sentence-transformers model avoids cloud embedding quotas and
    keeps query-time embeddings identical to the ones used to build the index.
    """
    logger.info("Loading embedding model: %s (device=%s)", EMBEDDING_MODEL, EMBEDDING_DEVICE)

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_docling_faiss_index(embeddings: HuggingFaceEmbeddings | None = None) -> FAISS:
    """Load the FAISS index built from Docling chunks."""
    if not os.path.exists(DOCLING_VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Docling FAISS index not found at {DOCLING_VECTORSTORE_DIR}. "
            "Please run build_docling_faiss_index.py first."
        )

    embeddings = embeddings or get_embeddings()
    logger.info("Loading FAISS index from %s", DOCLING_VECTORSTORE_DIR)

    return FAISS.load_local(
        DOCLING_VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
