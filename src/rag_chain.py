from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.config import GOOGLE_API_KEY, LLM_MODEL, RETRIEVER_K
from src.vector_store import load_faiss_index


_vectorstore = None
_llm = None


def load_prompt(prompt_path: str):
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = load_faiss_index()

    return _vectorstore


def get_llm():
    global _llm

    if _llm is None:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )

    return _llm


def format_docs(docs):
    formatted_context = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        content_type = doc.metadata.get("content_type", "text")
        table_title = doc.metadata.get("table_title", "")

        title_text = f", Table: {table_title}" if table_title else ""

        formatted_context.append(
            f"Source: {source}, Page: {page}, Type: {content_type}{title_text}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted_context)


def format_citations(docs):
    citation_map = {}

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        content_type = doc.metadata.get("content_type", "text")
        table_title = doc.metadata.get("table_title", "")

        key = f"{source}_page_{page}"

        if content_type == "table" and table_title:
            citation_map[key] = f"{source}, Page {page}, {table_title}"
        else:
            if key not in citation_map:
                citation_map[key] = f"{source}, Page {page}"

    return list(citation_map.values())


def format_chat_history(chat_history):
    """
    Converts Streamlit chat history into plain text for question reformulation.
    Expects list of dicts:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """

    if not chat_history:
        return "No previous conversation."

    formatted_messages = []

    for message in chat_history:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            formatted_messages.append(f"User: {content}")
        elif role == "assistant":
            formatted_messages.append(f"Assistant: {content}")

    return "\n".join(formatted_messages)


def reformulate_question(question: str, chat_history):
    """
    Converts follow-up question into standalone question.
    """

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

    standalone_question = response.content.strip()

    return standalone_question


def generate_answer(question: str, docs):
    """
    Generates final answer using retrieved documents.
    """

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


def retrieve_documents(question: str):
    """
    Retrieves relevant documents from FAISS.
    """

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )

    docs = retriever.invoke(question)

    return docs


def answer_question(question: str):
    """
    Basic RAG answer without chat history.
    Kept for test files and backward compatibility.
    """

    docs = retrieve_documents(question)
    answer = generate_answer(question, docs)
    citations = format_citations(docs)

    return {
        "answer": answer,
        "citations": citations,
        "source_documents": docs,
        "standalone_question": question
    }


def answer_question_with_history(question: str, chat_history):
    """
    History-aware RAG answer.
    Used by Streamlit app for follow-up questions.
    """

    standalone_question = reformulate_question(
        question=question,
        chat_history=chat_history
    )

    docs = retrieve_documents(standalone_question)
    answer = generate_answer(standalone_question, docs)
    citations = format_citations(docs)

    return {
        "answer": answer,
        "citations": citations,
        "source_documents": docs,
        "standalone_question": standalone_question
    }