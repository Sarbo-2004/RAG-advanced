import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR


def get_embeddings():
    """
    Load Hugging Face embedding model locally.
    This avoids Gemini API embedding quota limits.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


def create_faiss_index(chunks):
    """
    Create FAISS vector store from document chunks.
    """

    embeddings = get_embeddings()

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    print(f"Total chunks to embed: {len(chunks)}")
    print("Creating FAISS index using Hugging Face embeddings...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"FAISS index saved at: {VECTORSTORE_DIR}")

    return vectorstore


def load_faiss_index():
    """
    Load existing FAISS vector store.
    """

    embeddings = get_embeddings()

    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at {VECTORSTORE_DIR}. Please create the index first."
        )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore
