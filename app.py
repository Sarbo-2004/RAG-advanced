"""Professional Streamlit UI for FAO RAG Assistant."""

import uuid

import streamlit as st

from src.citations import clean_section_for_citation, format_citations, is_generic_section
from src.config import (
    LLM_MODEL,
    RERANK_TOP_K,
    RETRIEVER_K,
    validate_runtime_config,
)
from src.rag_service import RAGService


# --------------------------------------------------
# Constants
# --------------------------------------------------
ASSISTANT_AVATAR = "📘"
USER_AVATAR = "🧑‍💼"

CONTENT_TYPE_LABELS = {
    "docling_table_markdown": "Table",
    "docling_table": "Table",
    "docling_text": "Text",
    "table": "Table",
    "text": "Text",
}

EXAMPLE_QUESTIONS = [
    "What is true cost accounting?",
    "What are hidden costs in agrifood systems?",
    "What percentage of the world population was unable to afford a healthy diet in 2019?",
    "What are the three main ways to manage agrifood systems risk and uncertainty?",
    "How do monitoring requirements differ between land management and land-use change interventions?",
]

FOLLOW_UP_TESTS = [
    {
        "first": "What is true cost accounting?",
        "follow": "Why is it useful?",
    },
    {
        "first": "What are hidden costs in agrifood systems?",
        "follow": "Are they higher in low-income countries?",
    },
    {
        "first": "What percentage of the world population was unable to afford a healthy diet in 2019?",
        "follow": "How many people does that represent?",
    },
    {
        "first": "What are the three main ways to manage agrifood systems risk and uncertainty?",
        "follow": "Which one is linked with more predictable shocks?",
    },
    {
        "first": "How do monitoring requirements differ between land management and land-use change interventions?",
        "follow": "Which one can use remote sensing data?",
    },
]


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="FAO RAG Assistant",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .app-title {
            font-size: 1.75rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .app-subtitle {
            font-size: 0.92rem;
            color: #9ca3af;
            margin-bottom: 1.25rem;
        }

        .status-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1.25rem;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid rgba(127,127,127,0.25);
            background: rgba(127,127,127,0.10);
        }

        .green-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
        }

        .welcome-box {
            border: 1px solid rgba(127,127,127,0.20);
            background: rgba(127,127,127,0.05);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .welcome-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stButton"] button {
            border-radius: 10px;
            min-height: 2.6rem;
            font-size: 0.9rem;
        }

        div[data-testid="stChatMessage"] {
            padding: 0.45rem 0;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
        }

        .source-title {
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .source-meta {
            font-size: 0.82rem;
            opacity: 0.75;
            margin-bottom: 0.4rem;
        }

        .input-panel {
            border: 1px solid rgba(127,127,127,0.22);
            background: rgba(127,127,127,0.06);
            border-radius: 14px;
            padding: 0.75rem;
            margin-top: 1.25rem;
        }

        div[data-testid="stTextInput"] input {
            min-height: 46px;
            border-radius: 10px;
            font-size: 0.95rem;
        }

        .stFormSubmitButton button {
            min-height: 46px;
            border-radius: 10px;
            font-weight: 700;
        }

        pre {
            max-height: 240px !important;
            overflow-y: auto !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
            font-size: 0.78rem !important;
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


# --------------------------------------------------
# Cached RAG Service
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading models and FAISS index...")
def get_rag_service() -> RAGService:
    validate_runtime_config()
    return RAGService()


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def set_pending_question(question: str):
    st.session_state.pending_question = question


def render_sources(docs, citations):
    if not docs:
        return

    with st.expander(f"Sources and citations ({len(citations)})", expanded=False):
        if citations:
            st.markdown("**Citations**")
            for citation in citations:
                st.markdown(f"- {citation}")

        st.divider()
        st.markdown("**Retrieved passages**")

        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata

            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "Unknown page")
            year = metadata.get("year", "")
            section = metadata.get("section", "")
            content_type = metadata.get("content_type", "text")

            content_label = CONTENT_TYPE_LABELS.get(content_type, "Text")

            st.markdown(
                f"""
                <div class="source-title">{index}. {source}</div>
                <div class="source-meta">
                    Page {page}
                    {f" · {year}" if year else ""}
                    · {content_label}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if section and not is_generic_section(section):
                clean_section = clean_section_for_citation(section)
                st.caption(f"Section: {clean_section}")

            content = doc.page_content.strip()
            preview = content[:1200]

            if len(content) > 1200:
                preview += " ..."

            st.code(preview, language="text")


def stream_answer(service, question, session_id, captured):
    notice = captured["notice"]
    started = False

    for chunk in service.stream(question, session_id):
        if chunk.get("context"):
            captured["docs"] = chunk["context"]
            notice.info("Generating grounded answer...")

        token = chunk.get("answer")

        if token:
            if not started:
                notice.empty()
                started = True

            yield token


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## FAO RAG Assistant")
    st.caption("Conversational document assistant for FAO reports, 2021–2025.")

    st.divider()

    st.markdown("### Pipeline")
    st.write(f"**LLM:** {LLM_MODEL}")
    st.write("**Vector store:** FAISS")
    st.write("**Embeddings:** Local MiniLM")
    st.write("**Reranker:** Local Cross-Encoder")

    col_a, col_b = st.columns(2)
    col_a.metric("Retrieve", RETRIEVER_K)
    col_b.metric("Rerank", RERANK_TOP_K)

    st.divider()

    st.markdown("### Example questions")
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(question, key=f"example_{index}", use_container_width=True):
            set_pending_question(question)

    st.divider()

    st.markdown("### Follow-up tests")
    for index, pair in enumerate(FOLLOW_UP_TESTS):
        with st.expander(f"Test {index + 1}"):
            st.markdown("**First ask:**")
            st.code(pair["first"], language="text")
            st.markdown("**Then ask:**")
            st.code(pair["follow"], language="text")

    st.divider()

    turns = sum(1 for message in st.session_state.messages if message["role"] == "user")
    st.caption(f"{turns} question(s) in this session")

    if st.button("New conversation", type="primary", use_container_width=True):
        try:
            get_rag_service().reset_session(st.session_state.session_id)
        except Exception:
            pass

        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="app-title">📘 FAO RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask citation-backed questions from the FAO State of Food and Agriculture reports, 2021–2025.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status-row">
        <div class="status-chip"><span class="green-dot"></span>{LLM_MODEL}</div>
        <div class="status-chip">FAISS</div>
        <div class="status-chip">Local embeddings</div>
        <div class="status-chip">Cross-encoder reranking</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Empty State
# --------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-box">
            <div class="welcome-title">Start a conversation</div>
            Ask about concepts, tables, policy comparisons, or follow up naturally.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_1, col_2 = st.columns(2)

    for index, question in enumerate(EXAMPLE_QUESTIONS[:4]):
        target_col = col_1 if index % 2 == 0 else col_2

        if target_col.button(question, key=f"quick_{index}", use_container_width=True):
            set_pending_question(question)


# --------------------------------------------------
# Render Chat History
# --------------------------------------------------
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("citations", []),
            )


# --------------------------------------------------
# Input Panel
# --------------------------------------------------
with st.container(border=True):
    with st.form("chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([10, 1.2])

        with input_col:
            typed_question = st.text_input(
                label="Question",
                label_visibility="collapsed",
                placeholder="Message the FAO RAG assistant...",
            )

        with send_col:
            submitted = st.form_submit_button("Send", use_container_width=True)


if submitted and typed_question:
    user_question = typed_question
else:
    user_question = st.session_state.pending_question

st.session_state.pending_question = None


# --------------------------------------------------
# Handle New Message
# --------------------------------------------------
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
            notice.info("Retrieving and reranking relevant passages...")

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
            citations = format_citations(docs)

            render_sources(docs, citations)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "sources": docs,
                }
            )

            st.rerun()

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

            st.rerun()