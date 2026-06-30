"""Clean chatbot-style Streamlit UI for FAO RAG Assistant."""

import uuid
import pandas as pd
import streamlit as st

from src.citations import clean_section_for_citation, format_citations, is_generic_section
from src.config import validate_runtime_config
from src.rag_service import RAGService


# ==================================================
# Constants
# ==================================================
ASSISTANT_AVATAR = "📘"
USER_AVATAR = "🧑‍💼"

EXAMPLE_QUESTIONS = [
    "What is true cost accounting?",
    "What are hidden costs in agrifood systems?",
    "What percentage of the world population was unable to afford a healthy diet in 2019?",
    "What are the three main ways to manage agrifood systems risk and uncertainty?",
]

CONTENT_TYPE_LABELS = {
    "docling_table_markdown": "Table",
    "docling_table": "Table",
    "docling_figure_text": "Figure/Text",
    "docling_text": "Text",
    "table": "Table",
    "text": "Text",
}


# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="FAO RAG Assistant",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==================================================
# CSS
# ==================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 860px;
            padding-top: 2.25rem;
            padding-bottom: 6rem;
        }

        .chat-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .chat-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.35rem;
        }

        .chat-subtitle {
            font-size: 0.96rem;
            opacity: 0.68;
            line-height: 1.45;
            max-width: 680px;
            margin: 0 auto;
        }

        .starter-box {
            border: 1px solid rgba(127,127,127,0.18);
            background: rgba(127,127,127,0.045);
            border-radius: 16px;
            padding: 1.1rem;
            margin-bottom: 1rem;
        }

        .starter-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .starter-text {
            font-size: 0.9rem;
            opacity: 0.72;
        }

        div[data-testid="stButton"] button {
            border-radius: 12px;
            min-height: 2.75rem;
            font-size: 0.9rem;
            white-space: normal;
            border: 1px solid rgba(127,127,127,0.22);
        }

        div[data-testid="stChatMessage"] {
            padding: 0.38rem 0;
        }

        div[data-testid="stChatMessageContent"] {
            line-height: 1.58;
            font-size: 0.98rem;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
        }

        .source-title {
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }

        .source-meta {
            font-size: 0.78rem;
            opacity: 0.68;
            margin-bottom: 0.35rem;
        }

        pre {
            max-height: 230px !important;
            overflow-y: auto !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
            font-size: 0.76rem !important;
            border-radius: 10px !important;
        }

        footer {
            visibility: hidden;
        }

        #MainMenu {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Cached RAG Service
# ==================================================
@st.cache_resource(show_spinner="Loading FAO RAG Assistant...")
def get_rag_service() -> RAGService:
    validate_runtime_config()
    return RAGService()


# ==================================================
# Session State
# ==================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ==================================================
# Helper Functions
# ==================================================
def set_pending_question(question: str):
    st.session_state.pending_question = question


def safe_format_citations(docs, question: str = ""):
    try:
        return format_citations(docs, question)
    except TypeError:
        return format_citations(docs)


def preview_text(text: str, limit: int = 1200) -> str:
    text = text.strip()

    if len(text) > limit:
        return text[:limit] + " ..."

    return text


def is_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pipe_lines = [line for line in lines if "|" in line]

    if len(pipe_lines) < 2:
        return False

    separator_lines = [
        line for line in pipe_lines
        if set(line.replace("|", "").strip()) <= set("-: ")
        and "-" in line
    ]

    return len(separator_lines) >= 1


def make_unique_column_names(columns):
    unique_columns = []
    seen = {}

    for index, column in enumerate(columns, start=1):
        column_name = str(column).strip()

        if not column_name:
            column_name = f"Column {index}"

        if column_name in seen:
            seen[column_name] += 1
            column_name = f"{column_name}_{seen[column_name]}"
        else:
            seen[column_name] = 1

        unique_columns.append(column_name)

    return unique_columns


def markdown_table_to_dataframe(text: str):
    """
    Converts a Markdown table into a pandas DataFrame safely.

    Handles:
    - blank column names
    - duplicate column names
    - uneven row lengths
    - messy Docling table extraction
    """

    lines = [line.strip() for line in text.splitlines() if "|" in line]

    if len(lines) < 2:
        return None

    normalized_lines = [
        line.strip().strip("|")
        for line in lines
    ]

    data_lines = []

    for line in normalized_lines:
        cells = [cell.strip() for cell in line.split("|")]

        separator_check = "".join(cells)
        separator_check = separator_check.replace("-", "")
        separator_check = separator_check.replace(":", "")
        separator_check = separator_check.strip()

        # Skip Markdown separator row like --- | --- | ---
        if separator_check:
            data_lines.append(cells)

    if len(data_lines) < 2:
        return None

    raw_header = data_lines[0]
    rows = data_lines[1:]

    header = make_unique_column_names(raw_header)
    max_cols = len(header)

    cleaned_rows = []

    for row in rows:
        row = [str(cell).strip() for cell in row]

        if len(row) < max_cols:
            row = row + [""] * (max_cols - len(row))
        elif len(row) > max_cols:
            row = row[:max_cols]

        cleaned_rows.append(row)

    try:
        dataframe = pd.DataFrame(cleaned_rows, columns=header)
        return dataframe
    except Exception:
        return None


def render_sources(docs, citations):
    if not docs:
        return

    with st.expander(f"📚 Sources and citations ({len(citations)})", expanded=False):
        if citations:
            st.markdown("**Citations**")
            for citation in citations:
                st.markdown(f"- {citation}")

        st.divider()
        st.markdown("**Retrieved context preview**")

        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata

            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "Unknown page")
            year = metadata.get("year", "")
            section = metadata.get("section", "")
            content_type = metadata.get("content_type", "text")
            label = CONTENT_TYPE_LABELS.get(content_type, "Text")

            st.markdown(
                f"""
                <div class="source-title">{index}. {source}</div>
                <div class="source-meta">Page {page}{f" · {year}" if year else ""} · {label}</div>
                """,
                unsafe_allow_html=True,
            )

            if section and not is_generic_section(section):
                st.caption(f"Section: {clean_section_for_citation(section)}")

            content = doc.page_content.strip()

            if is_markdown_table(content):
                dataframe = markdown_table_to_dataframe(content)

                if dataframe is not None and not dataframe.empty:
                    try:
                        st.dataframe(
                            dataframe,
                            use_container_width=True,
                            hide_index=True,
                        )
                    except Exception:
                        st.code(preview_text(content), language="text")
                else:
                    st.code(preview_text(content), language="text")
            else:
                st.code(preview_text(content), language="text")


def stream_answer(service, question, session_id, captured):
    notice = captured["notice"]
    started = False

    for chunk in service.stream(question, session_id):
        if chunk.get("context"):
            captured["docs"] = chunk["context"]
            notice.info("Reviewing retrieved evidence...")

        token = chunk.get("answer")

        if token:
            if not started:
                notice.empty()
                started = True

            yield token


# ==================================================
# Header
# ==================================================
st.markdown(
    """
    <div class="chat-header">
        <div class="chat-title">📘 FAO RAG Assistant</div>
        <div class="chat-subtitle">
            Ask questions from the FAO State of Food and Agriculture reports, 2021–2025.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Empty State
# ==================================================
if not st.session_state.messages:
    st.markdown(
        """
        <div class="starter-box">
            <div class="starter-title">How can I help?</div>
            <div class="starter-text">
                Ask a question about the FAO reports or choose an example below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q_col_1, q_col_2 = st.columns(2)

    for index, question in enumerate(EXAMPLE_QUESTIONS):
        target_col = q_col_1 if index % 2 == 0 else q_col_2

        if target_col.button(
            question,
            key=f"example_{index}",
            use_container_width=True,
        ):
            set_pending_question(question)


# ==================================================
# Render Chat History
# ==================================================
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("citations", []),
            )


# ==================================================
# Chat Input
# ==================================================
typed_question = st.chat_input("Ask a question about the FAO reports...")

if typed_question:
    user_question = typed_question
else:
    user_question = st.session_state.pending_question

st.session_state.pending_question = None


# ==================================================
# New Conversation Button
# ==================================================
if st.session_state.messages:
    if st.button("Start new conversation"):
        try:
            get_rag_service().reset_session(st.session_state.session_id)
        except Exception:
            pass

        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()


# ==================================================
# Handle New Question
# ==================================================
if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            service = get_rag_service()

            notice = st.empty()
            notice.info("Retrieving and reranking relevant evidence...")

            captured = {
                "notice": notice,
                "docs": [],
            }

            answer = st.write_stream(
                stream_answer(
                    service=service,
                    question=user_question,
                    session_id=st.session_state.session_id,
                    captured=captured,
                )
            )

            docs = captured.get("docs", [])
            citations = safe_format_citations(docs, user_question)

            render_sources(docs, citations)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "sources": docs,
                }
            )

        except Exception as error:
            error_message = (
                "Something went wrong while generating the answer. "
                "Please check the FAISS index, API key, or local model paths."
            )

            st.error(error_message)

            with st.expander("Error details"):
                st.code(str(error), language="text")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "citations": [],
                    "sources": [],
                }
            )