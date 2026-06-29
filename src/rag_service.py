"""Conversational RAG service built from native LangChain primitives.

This module wires together the three mandated LangChain pillars:

* Retrievers  -> reranking retriever from ``src.retriever``
* Chains      -> ``create_history_aware_retriever`` + ``create_stuff_documents_chain``
                 composed with ``create_retrieval_chain`` (LCEL under the hood)
* Memory      -> ``RunnableWithMessageHistory`` backed by per-session
                 ``ChatMessageHistory`` objects

The public surface is a single :class:`RAGService` whose ``answer`` /
``stream`` methods take a ``session_id`` so each chat keeps its own memory.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

from src.citations import format_citations
from src.config import (
    ANSWER_TEMPERATURE,
    CONTEXTUALIZE_PROMPT_PATH,
    GOOGLE_API_KEY,
    LLM_MODEL,
    QA_PROMPT_PATH,
    get_logger,
)
from src.retriever import build_reranking_retriever

logger = get_logger(__name__)


def _load_prompt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def _build_llm() -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    logger.info("Initialising Gemini LLM: %s (temperature=%s)", LLM_MODEL, ANSWER_TEMPERATURE)
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=ANSWER_TEMPERATURE,
    )


class RAGService:
    """Stateful conversational RAG service.

    One instance is created per process and reused across requests; conversation
    state is isolated per ``session_id`` via the message-history store.
    """

    def __init__(self) -> None:
        self._store: Dict[str, ChatMessageHistory] = {}

        llm = _build_llm()
        retriever = build_reranking_retriever()

        # --- Chain 1: history-aware retrieval -------------------------------
        # Reformulates the latest question into a standalone query using chat
        # history, then runs the reranking retriever.
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _load_prompt_text(CONTEXTUALIZE_PROMPT_PATH)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        # --- Chain 2: grounded answer generation ----------------------------
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _load_prompt_text(QA_PROMPT_PATH)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # --- Compose: retrieve-then-answer ----------------------------------
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # --- Memory: per-session chat history -------------------------------
        self._conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        logger.info("RAGService ready.")

    # ------------------------------------------------------------------ memory
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def reset_session(self, session_id: str) -> None:
        """Drop all stored memory for a session."""
        self._store.pop(session_id, None)

    # ------------------------------------------------------------------ public
    def answer(self, question: str, session_id: str) -> Dict[str, Any]:
        """Answer a question within a conversation, returning answer + sources."""
        logger.info("[session=%s] Q: %s", session_id, question)

        result = self._conversational_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )

        documents: List[Document] = result.get("context", [])
        return {
            "answer": result.get("answer", ""),
            "citations": format_citations(documents),
            "source_documents": documents,
        }

    def stream(self, question: str, session_id: str) -> Iterator[Dict[str, Any]]:
        """Stream the answer. Yields dicts that may contain ``answer`` token
        deltas and (once) the retrieved ``context`` documents."""
        logger.info("[session=%s] Q (stream): %s", session_id, question)

        for chunk in self._conversational_chain.stream(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        ):
            yield chunk
