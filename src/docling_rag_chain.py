from sentence_transformers import CrossEncoder

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.docling_vector_store import load_docling_faiss_index

from src.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    RETRIEVER_K,
    RERANK_TOP_K,
    RERANKER_MODEL,
    MAX_HISTORY_MESSAGES,
    ANSWER_TEMPERATURE
)


_docling_vectorstore = None
_llm = None
_reranker = None


# --------------------------------------------------
# Prompt Loader
# --------------------------------------------------
def load_prompt(prompt_path: str):
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


# --------------------------------------------------
# Cached FAISS Loader
# --------------------------------------------------
def get_docling_vectorstore():
    global _docling_vectorstore

    if _docling_vectorstore is None:
        _docling_vectorstore = load_docling_faiss_index()

    return _docling_vectorstore


# --------------------------------------------------
# Cached Gemini LLM
# --------------------------------------------------
def get_llm():
    global _llm

    if _llm is None:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=ANSWER_TEMPERATURE
        )

    return _llm


# --------------------------------------------------
# Cached CrossEncoder Reranker
# --------------------------------------------------
def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)

    return _reranker


# --------------------------------------------------
# Citation Helpers
# --------------------------------------------------
def is_generic_section(section: str) -> bool:
    if not section:
        return True

    section_clean = section.strip().lower()
    section_without_hash = section_clean.lstrip("#").strip()

    generic_sections = [
        "untitled section",
        "the state of food and agriculture 2021 in brief",
        "the state of food and agriculture 2022 in brief",
        "the state of food and agriculture 2023 in brief",
        "the state of food and agriculture 2024 in brief",
        "the state of food and agriculture 2025 in brief",
        "required citation:",
        "food and agriculture the state of"
    ]

    return section_without_hash in generic_sections


def clean_section_for_citation(section: str) -> str:
    if not section:
        return ""

    section = section.strip()
    section = section.lstrip("#").strip()

    return section


def format_citations(docs):
    citation_map = {}

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        section = doc.metadata.get("section", "")
        content_type = doc.metadata.get("content_type", "")

        cleaned_section = clean_section_for_citation(section)

        key = f"{source}_page_{page}_{cleaned_section}_{content_type}"

        if cleaned_section and not is_generic_section(section):
            citation = f"{source}, Page {page}, Section: {cleaned_section}"
        else:
            citation = f"{source}, Page {page}"

        if key not in citation_map:
            citation_map[key] = citation

    return list(citation_map.values())


# --------------------------------------------------
# Context Formatting
# --------------------------------------------------
def format_docs(docs):
    formatted_context = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        section = doc.metadata.get("section", "")
        content_type = doc.metadata.get("content_type", "docling_text")
        parser = doc.metadata.get("parser", "docling")
        table_structure = doc.metadata.get("table_structure", False)

        section_text = clean_section_for_citation(section)

        formatted_context.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Section: {section_text}\n"
            f"Content Type: {content_type}\n"
            f"Parser: {parser}\n"
            f"Table Structure Enabled: {table_structure}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted_context)


# --------------------------------------------------
# FAISS Retrieval
# --------------------------------------------------
def retrieve_docling_documents(question: str):
    vectorstore = get_docling_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )

    docs = retriever.invoke(question)

    return docs


# --------------------------------------------------
# CrossEncoder Reranking
# --------------------------------------------------
def rerank_documents(question: str, docs):
    """
    CrossEncoder reranking.
    FAISS retrieves candidate chunks first.
    CrossEncoder then reranks them based on query-document relevance.
    """

    if not docs:
        return []

    reranker = get_reranker()

    pairs = [
        [question, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs = sorted(
        scored_docs,
        key=lambda item: item[1],
        reverse=True
    )

    reranked_docs = [
        doc for doc, score in scored_docs[:RERANK_TOP_K]
    ]

    return reranked_docs


def retrieve_and_rerank_docling_documents(question: str):
    initial_docs = retrieve_docling_documents(question)
    reranked_docs = rerank_documents(question, initial_docs)

    return reranked_docs


# --------------------------------------------------
# Answer Generation
# --------------------------------------------------
def generate_answer(question: str, docs):
    context = format_docs(docs)

    prompt_text = load_prompt("prompts/qa_prompt.txt")

    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["context", "question"]
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    llm = get_llm()
    response = llm.invoke(final_prompt)

    return response.content


def answer_question_docling(question: str):
    docs = retrieve_and_rerank_docling_documents(question)
    answer = generate_answer(question, docs)
    citations = format_citations(docs)

    return {
        "answer": answer,
        "citations": citations,
        "source_documents": docs,
        "standalone_question": question
    }


# --------------------------------------------------
# Conversation Memory / Reformulation
# --------------------------------------------------
def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."

    recent_history = chat_history[-MAX_HISTORY_MESSAGES:]

    formatted_messages = []

    for message in recent_history:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            formatted_messages.append(f"User: {content}")
        elif role == "assistant":
            formatted_messages.append(f"Assistant: {content}")

    return "\n".join(formatted_messages)


def reformulate_question(question: str, chat_history):
    if not chat_history:
        return question

    llm = get_llm()

    prompt_text = load_prompt("prompts/contextualize_prompt.txt")

    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["chat_history", "question"]
    )

    formatted_prompt = prompt.format(
        chat_history=format_chat_history(chat_history),
        question=question
    )

    response = llm.invoke(formatted_prompt)

    return response.content.strip()


def answer_question_docling_with_history(question: str, chat_history):
    standalone_question = reformulate_question(
        question=question,
        chat_history=chat_history
    )

    docs = retrieve_and_rerank_docling_documents(standalone_question)
    answer = generate_answer(standalone_question, docs)
    citations = format_citations(docs)

    return {
        "answer": answer,
        "citations": citations,
        "source_documents": docs,
        "standalone_question": standalone_question
    }