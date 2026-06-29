"""Integration tests for the conversational RAG service.

These hit the real FAISS index, reranker, and Gemini API, so they are skipped
unless GOOGLE_API_KEY is set and the index exists. Run with:

    pytest tests/test_rag_integration.py -v
"""

import os

import pytest

from src.config import DOCLING_VECTORSTORE_DIR

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY") or not os.path.exists(DOCLING_VECTORSTORE_DIR),
    reason="Requires GOOGLE_API_KEY and a built FAISS index.",
)


@pytest.fixture(scope="module")
def service():
    from src.rag_service import RAGService

    return RAGService()


def test_grounded_answer_with_citations(service):
    result = service.answer("What is true cost accounting?", session_id="test-1")

    assert result["answer"].strip()
    assert result["source_documents"], "expected retrieved documents"
    assert result["citations"], "expected at least one citation"


def test_multi_turn_memory_resolves_pronoun(service):
    session = "test-2"
    service.answer("What is true cost accounting?", session_id=session)

    # The follow-up only makes sense if conversation memory is working.
    follow_up = service.answer("Which report discusses it?", session_id=session)

    assert follow_up["answer"].strip()
    assert "could not find" not in follow_up["answer"].lower()


def test_sessions_are_isolated(service):
    service.answer("What are hidden costs in agrifood systems?", session_id="iso-a")
    service.reset_session("iso-a")
    # A fresh session has no memory; an ambiguous pronoun should not resolve.
    assert service._get_session_history("iso-a").messages == []
