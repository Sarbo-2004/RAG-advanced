"""Embedding model and FAISS index loading for the Docling pipeline."""

import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    DOCLING_VECTORSTORE_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local Hugging Face embedding model.

    This uses the same local sentence-transformers model that was used
    while building the FAISS index.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_docling_faiss_index(
    embeddings: HuggingFaceEmbeddings | None = None
) -> FAISS:
    """Load the FAISS index built from Docling chunks."""

    if not os.path.exists(DOCLING_VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Docling FAISS index not found at {DOCLING_VECTORSTORE_DIR}. "
            "Please run build_docling_faiss_index.py first."
        )

    embeddings = embeddings or get_embeddings()

    return FAISS.load_local(
        DOCLING_VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )