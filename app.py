import streamlit as st

from src.rag_chain import answer_question_with_history
from src.docling_rag_chain import answer_question_docling_with_history


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="FAO PDF RAG Chatbot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Minimal Theme-Safe CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }

        .app-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            font-size: 0.95rem;
            color: #9ca3af;
            margin-bottom: 1.5rem;
        }

        .source-meta {
            font-size: 0.85rem;
            color: #9ca3af;
        }

        .small-note {
            font-size: 0.85rem;
            color: #9ca3af;
        }

        div[data-testid="stSidebar"] button {
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_standalone_question" not in st.session_state:
    st.session_state.last_standalone_question = ""

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="app-title">📘 FAO PDF RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Conversational RAG over FAO reports using Gemini, FAISS, local Hugging Face embeddings, and Docling enterprise parsing.</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")

    retrieval_mode = st.radio(
        "Retrieval Mode",
        ["Docling Enterprise Pipeline", "Classic Pipeline"],
        index=0
    )

    if retrieval_mode == "Docling Enterprise Pipeline":
        st.success("Docling Enterprise Mode Active")
        st.caption("Page-wise Docling extraction, Markdown-aware chunking, reranking, and section-level citations.")
    else:
        st.info("Classic Pipeline Mode Active")
        st.caption("PyMuPDF text extraction, pdfplumber tables, and classic recursive chunking.")

    st.divider()

    st.subheader("📚 Dataset")
    st.markdown(
        """
        **Documents:** FAO Reports 2021–2025  
        **LLM:** Gemini  
        **Vector Store:** FAISS  
        **Embeddings:** Local MiniLM  
        **Tables:** Table-aware retrieval  
        """
    )

    st.divider()

    st.subheader("🧪 Example Questions")

    example_questions = [
        "What is true cost accounting?",
        "What are hidden costs in agrifood systems?",
        "Are hidden costs higher in low-income countries?",
        "According to Table 2 in the 2021 report, what percentage of the world population was unable to afford a healthy diet in 2019?",
        "According to Table 5 in the 2021 report, what are the three main ways to manage agrifood systems risk and uncertainty?",
        "According to Table 3 in the 2025 report, how do monitoring requirements differ between land management and land-use change interventions?"
    ]

    for question in example_questions:
        if st.button(question, use_container_width=True):
            st.session_state.selected_question = question

    st.divider()

    st.subheader("🧠 Reformulated Query")

    if st.session_state.last_standalone_question:
        st.info(st.session_state.last_standalone_question)
    else:
        st.caption("The standalone version of the latest question will appear here.")

    st.divider()

    st.subheader("🔎 Retrieved Sources")

    if st.session_state.last_sources:
        for index, doc in enumerate(st.session_state.last_sources, start=1):
            metadata = doc.metadata

            source = metadata.get("source", "Unknown Source")
            page = metadata.get("page", "Unknown Page")
            content_type = metadata.get("content_type", "text")
            section = metadata.get("section", "")
            table_title = metadata.get("table_title", "")
            parser = metadata.get("parser", "")

            with st.expander(f"{index}. {source} | Page {page}"):
                st.markdown(f"**Content Type:** `{content_type}`")

                if parser:
                    st.markdown(f"**Parser:** `{parser}`")

                if section:
                    st.markdown(f"**Section:** {section}")

                if table_title:
                    st.markdown(f"**Table:** {table_title}")

                st.markdown("**Snippet:**")
                st.code(doc.page_content[:1200], language="text")
    else:
        st.caption("Retrieved source chunks will appear here after a question is asked.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.last_standalone_question = ""
            st.session_state.selected_question = None
            st.rerun()

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()


# --------------------------------------------------
# Welcome Message
# --------------------------------------------------
if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("### 👋 Welcome")
        st.markdown(
            """
            Ask questions from the FAO reports.  
            You can ask:
            
            - Conceptual questions  
            - Table-specific questions  
            - Follow-up questions  
            - Report/year-specific questions  
            
            Example:
            ```text
            What is true cost accounting?
            ```
            Then follow up with:
            ```text
            Which report discusses it?
            ```
            """
        )


# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            citations = message.get("citations", [])

            if citations:
                with st.expander("📌 Sources"):
                    for citation in citations:
                        st.markdown(f"- {citation}")


# --------------------------------------------------
# Chat Input
# --------------------------------------------------
typed_question = st.chat_input("Ask a question from the FAO PDF reports...")

user_question = typed_question or st.session_state.selected_question


# --------------------------------------------------
# Handle Query
# --------------------------------------------------
if user_question:
    st.session_state.selected_question = None

    previous_chat_history = st.session_state.messages.copy()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                if retrieval_mode == "Classic Pipeline":
                    result = answer_question_with_history(
                        question=user_question,
                        chat_history=previous_chat_history
                    )
                else:
                    result = answer_question_docling_with_history(
                        question=user_question,
                        chat_history=previous_chat_history
                    )

                answer = result.get("answer", "")
                citations = result.get("citations", [])
                source_documents = result.get("source_documents", [])
                standalone_question = result.get("standalone_question", user_question)

                st.markdown(answer)

                if citations:
                    with st.expander("📌 Sources"):
                        for citation in citations:
                            st.markdown(f"- {citation}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    }
                )

                st.session_state.last_sources = source_documents
                st.session_state.last_standalone_question = standalone_question

            except Exception as e:
                error_message = (
                    "⚠️ Something went wrong while generating the answer. "
                    "Please check the selected retrieval mode, FAISS index, API key, or local model paths."
                )

                st.error(error_message)

                with st.expander("Error Details"):
                    st.code(str(e), language="text")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "citations": []
                    }
                )