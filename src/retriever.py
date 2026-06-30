"""LangChain retriever construction: FAISS similarity search wrapped in a
cross-encoder reranking compressor.

This builds the retriever layer:

FAISS retrieves RETRIEVER_K candidate chunks.
CrossEncoder reranker keeps the best RERANK_TOP_K chunks.
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from src.config import RERANK_TOP_K, RERANKER_MODEL, RETRIEVER_K
from src.vector_store import load_docling_faiss_index


def build_reranking_retriever(vectorstore=None) -> BaseRetriever:
    """Build a FAISS + CrossEncoder reranking retriever.

    Step 1:
        FAISS retrieves RETRIEVER_K candidate chunks.

    Step 2:
        CrossEncoder reranks those chunks using query-document relevance.

    Step 3:
        Only RERANK_TOP_K best chunks are passed to the RAG chain.
    """

    vectorstore = vectorstore or load_docling_faiss_index()

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL
    )

    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=RERANK_TOP_K
    )

    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    return reranking_retriever