"""Streamlit conversational RAG chatbot over the FAO 'State of Food and
Agriculture' reports (2021-2025).

UI layer only: all retrieval/memory/generation lives in ``src.rag_service``.
"""

import uuid

import streamlit as st

from src.citations import clean_section_for_citation, format_citations, is_generic_section
from src.config import (
    LLM_MODEL,
    RERANK_TOP_K,
    RETRIEVER_K,
    get_logger,
    validate_runtime_config,
)
from src.rag_service import RAGService

logger = get_logger(__name__)

ASSISTANT_AVATAR = "📘"
USER_AVATAR = "🧑‍💼"

# content_type -> (label, icon)
CONTENT_TYPE_BADGES = {
    "docling_table_markdown": ("Table", "📊"),
    "docling_table": ("Table", "📊"),
    "docling_text": ("Text", "📄"),
    "table": ("Table", "📊"),
    "text": ("Text", "📄"),
}

EXAMPLE_QUESTIONS = [
    "What is true cost accounting?",
    "What are hidden costs in agrifood systems?",
    "Are hidden costs higher in low-income countries?",
    "According to Table 2 in the 2021 report, what percentage of the world population was unable to afford a healthy diet in 2019?",
    "According to Table 5 in the 2021 report, what are the three main ways to manage agrifood systems risk and uncertainty?",
    "According to Table 3 in the 2025 report, how do monitoring requirements differ between land management and land-use change interventions?",
]


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="FAO RAG Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Theme-safe styling (works in both light and dark themes)
# --------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --accent: #10b981;
            --accent-2: #3b82f6;
        }

        .block-container { padding-top: 2.2rem; max-width: 1100px; }

        /* Header */
        .app-header {
            display: flex; align-items: center; justify-content: space-between;
            gap: 1rem; margin-bottom: 0.25rem;
        }
        .app-title {
            font-size: 2rem; font-weight: 800; line-height: 1.1;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .app-subtitle { font-size: 0.95rem; opacity: 0.7; margin: 0.3rem 0 1.2rem; }

        /* Status pill */
        .status-pill {
            display: inline-flex; align-items: center; gap: 0.45rem;
            padding: 0.32rem 0.7rem; border-radius: 999px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.35);
            font-size: 0.8rem; font-weight: 600; white-space: nowrap;
        }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--accent); box-shadow: 0 0 0 0 rgba(16,185,129,0.6);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
            70% { box-shadow: 0 0 0 7px rgba(16,185,129,0); }
            100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
        }

        /* Badges */
        .badge {
            display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 600; margin-right: 0.35rem;
            background: rgba(127,127,127,0.16); border: 1px solid rgba(127,127,127,0.22);
        }
        .badge-accent {
            background: rgba(16, 185, 129, 0.14);
            border-color: rgba(16, 185, 129, 0.3);
        }

        /* Buttons: example chips + controls */
        div[data-testid="stSidebar"] .stButton > button {
            text-align: left; border-radius: 10px;
            border: 1px solid rgba(127,127,127,0.2);
            transition: border-color 0.15s ease, transform 0.05s ease;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--accent);
        }
        .stButton > button:active { transform: translateY(1px); }

        /* Source snippet code blocks a touch tighter */
        .source-section { font-size: 0.82rem; opacity: 0.75; margin: 0.1rem 0 0.4rem; }

        /* Wide retrieved tables: scroll instead of clipping */
        [data-testid="stMarkdownContainer"] table {
            display: block; overflow-x: auto; white-space: nowrap;
            font-size: 0.82rem; border-collapse: collapse;
        }
        [data-testid="stMarkdownContainer"] th,
        [data-testid="stMarkdownContainer"] td {
            border: 1px solid rgba(127,127,127,0.25); padding: 0.3rem 0.55rem;
        }
        [data-testid="stMarkdownContainer"] thead th {
            background: rgba(16, 185, 129, 0.12);
        }

        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Cached, process-wide RAG service
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading models and FAISS index…")
def get_rag_service() -> RAGService:
    """Build the RAG service once and reuse it across reruns and sessions."""
    validate_runtime_config()
    return RAGService()


# --------------------------------------------------
# Per-browser-session state
# --------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, citations, sources}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _looks_like_markdown_table(text: str) -> bool:
    """Heuristic: at least two pipe rows and a separator row (---)."""
    pipe_lines = [line for line in text.splitlines() if line.count("|") >= 2]
    has_separator = any(set(line.strip()) <= set("|-: ") and "-" in line for line in pipe_lines)
    return len(pipe_lines) >= 2 and has_separator


def render_sources(docs, citations):
    """Render a polished, badge-rich source panel for an assistant turn."""
    if not docs:
        return

    with st.expander(f"📚 Sources & citations ({len(docs)})", expanded=False):
        if citations:
            st.markdown("**Citations**")
            for citation in citations:
                st.markdown(f"- {citation}")
            st.divider()

        st.markdown("**Retrieved passages**")
        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "?")
            year = metadata.get("year", "")
            section = metadata.get("section", "")
            content_type = metadata.get("content_type", "text")

            label, icon = CONTENT_TYPE_BADGES.get(content_type, ("Text", "📄"))

            header = (
                f"**{index}. {icon} {source}** "
                f"<span class='badge badge-accent'>Page {page}</span>"
                f"<span class='badge'>{label}</span>"
            )
            if year:
                header += f"<span class='badge'>{year}</span>"
            st.markdown(header, unsafe_allow_html=True)

            if section and not is_generic_section(section):
                st.markdown(
                    f"<div class='source-section'>§ {clean_section_for_citation(section)}</div>",
                    unsafe_allow_html=True,
                )

            content = doc.page_content
            with st.container(border=True):
                if _looks_like_markdown_table(content):
                    # Render the full table as a real (scrollable) table so the
                    # answer's evidence is actually visible, not cut off.
                    st.markdown(content, unsafe_allow_html=False)
                else:
                    snippet = content[:1100].strip()
                    if len(content) > 1100:
                        snippet += " …"
                    st.markdown(snippet)


def answer_token_stream(service, question, session_id, captured):
    """Yield answer tokens while capturing retrieved context as it streams."""
    notice = captured["notice"]
    started = False
    for chunk in service.stream(question, session_id):
        if chunk.get("context"):
            captured["docs"] = chunk["context"]
            notice.markdown("✍️ &nbsp;*Generating grounded answer…*", unsafe_allow_html=True)
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
    st.markdown("### 📘 FAO RAG Assistant")
    st.caption("Conversational retrieval over FAO reports, 2021–2025.")

    st.divider()

    with st.expander("⚙️ Pipeline & settings", expanded=True):
        st.markdown(
            f"""
            <span class='badge badge-accent'>Gemini</span> {LLM_MODEL}<br>
            <span class='badge'>Vector store</span> FAISS<br>
            <span class='badge'>Embeddings</span> MiniLM (local)<br>
            <span class='badge'>Reranker</span> Cross-encoder
            """,
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        col_a.metric("Retrieve k", RETRIEVER_K)
        col_b.metric("Rerank top-k", RERANK_TOP_K)

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            """
            1. **Reformulate** the question using chat history (memory).
            2. **Retrieve** candidate passages from FAISS.
            3. **Rerank** them with a cross-encoder.
            4. **Generate** a grounded answer with citations.

            Built with LangChain *Retrievers, Memory & Chains*.
            """
        )

    st.divider()

    st.markdown("#### 💡 Try an example")
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(question, use_container_width=True, key=f"side_q_{i}"):
            st.session_state.pending_question = question

    st.divider()

    turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.caption(f"💬 {turns} question(s) this session")
    if st.button("🧹 New conversation", use_container_width=True, type="primary"):
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
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <div class="app-title">📘 FAO PDF RAG Assistant</div>
        </div>
        <div class="status-pill"><span class="status-dot"></span> {LLM_MODEL}</div>
    </div>
    <div class="app-subtitle">
        Ask grounded, citation-backed questions about the FAO “State of Food and
        Agriculture” reports (2021–2025) — with natural multi-turn follow-ups.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Empty state: welcome + suggestion grid
# --------------------------------------------------
if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("#### 👋 Welcome")
        st.markdown(
            "Ask conceptual, table-specific, or year-specific questions and follow "
            "up naturally. The assistant remembers the conversation and grounds every "
            "answer in the source PDFs."
        )
        st.markdown("**Get started:**")
        grid = st.columns(2)
        for i, question in enumerate(EXAMPLE_QUESTIONS[:4]):
            if grid[i % 2].button(question, use_container_width=True, key=f"welcome_q_{i}"):
                st.session_state.pending_question = question


# --------------------------------------------------
# Render chat history
# --------------------------------------------------
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []), message.get("citations", []))


# --------------------------------------------------
# Resolve input (typed or example chip)
# --------------------------------------------------
typed_question = st.chat_input("Ask a question about the FAO reports…")
user_question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None


# --------------------------------------------------
# Handle a new question
# --------------------------------------------------
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            service = get_rag_service()

            notice = st.empty()
            notice.markdown("🔍 &nbsp;*Retrieving and reranking relevant passages…*", unsafe_allow_html=True)
            captured = {"notice": notice, "docs": []}

            answer = st.write_stream(
                answer_token_stream(service, user_question, st.session_state.session_id, captured)
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

        except Exception as error:  # surface a friendly message, keep details
            logger.exception("Failed to answer question")
            error_message = (
                "⚠️ Something went wrong while generating the answer. "
                "Please check the FAISS index, API key, or model paths."
            )
            st.error(error_message)
            with st.expander("Error details"):
                st.code(str(error), language="text")
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "citations": [], "sources": []}
            )
