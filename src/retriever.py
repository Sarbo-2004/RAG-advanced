"""LangChain retriever construction: FAISS MMR retrieval with cross-encoder reranking."""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from src.config import RERANK_TOP_K, RERANKER_MODEL, RETRIEVER_K
from src.vector_store import load_docling_faiss_index


def build_reranking_retriever(vectorstore=None) -> BaseRetriever:
    """Build FAISS MMR retriever + CrossEncoder reranker."""

    vectorstore = vectorstore or load_docling_faiss_index()

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": 50,
            "lambda_mult": 0.45,
        },
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL
    )

    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=RERANK_TOP_K
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )