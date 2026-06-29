import os
from langchain_community.vectorstores import FAISS
from src.vector_store import get_embeddings
from src.config import DOCLING_VECTORSTORE_DIR


def load_docling_faiss_index():
    """
    Load FAISS index created from Docling chunks.
    """

    embeddings = get_embeddings()

    if not os.path.exists(DOCLING_VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Docling FAISS index not found at {DOCLING_VECTORSTORE_DIR}. "
            "Please run build_docling_faiss_index.py first."
        )

    vectorstore = FAISS.load_local(
        DOCLING_VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore