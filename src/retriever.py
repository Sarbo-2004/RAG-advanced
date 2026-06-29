"""LangChain retriever construction: FAISS similarity search wrapped in a
cross-encoder reranking compressor.

This is the "Retrievers" pillar of the assignment expressed natively:
a base vector retriever composed with a ``ContextualCompressionRetriever`` so
reranking is part of the retriever interface rather than a manual post-step.
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from src.config import RERANK_TOP_K, RERANKER_MODEL, RETRIEVER_K, get_logger
from src.vector_store import load_docling_faiss_index

logger = get_logger(__name__)


def build_reranking_retriever(vectorstore=None) -> BaseRetriever:
    """Build a reranking retriever.

    FAISS retrieves ``RETRIEVER_K`` candidate chunks by vector similarity; a
    cross-encoder then rescores each (query, chunk) pair and keeps the top
    ``RERANK_TOP_K``. This two-stage design gives recall from the bi-encoder and
    precision from the cross-encoder.
    """
    vectorstore = vectorstore or load_docling_faiss_index()

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    logger.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=RERANK_TOP_K)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
