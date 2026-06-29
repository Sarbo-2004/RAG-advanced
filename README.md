# FAO PDF RAG Chatbot

A conversational Retrieval-Augmented Generation (RAG) chatbot over the FAO
*State of Food and Agriculture* reports (2021–2025). Ask questions about
policies, concepts, and tables and follow up naturally — the assistant keeps
conversation context and answers are grounded in the source PDFs with
section-level citations.

## Stack

| Layer | Choice |
|-------|--------|
| UI | Streamlit (chat) |
| Orchestration | LangChain (Chains, Memory, Retrievers) |
| LLM | Google Gemini (`langchain-google-genai`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | FAISS |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF parsing | Docling (page-wise, table-structure aware) |

## Architecture

```
PDFs ──(Docling page-wise parse)──► Markdown ──(heading-aware split)──► chunks
                                                                          │
                                                              embeddings ─┘
                                                                          ▼
                                                                    FAISS index
Query ─┐
       ▼
RunnableWithMessageHistory (Memory, per session)
       │
       ▼
create_retrieval_chain
   ├─ create_history_aware_retriever ── reformulate question ──► ContextualCompressionRetriever
   │                                                                ├─ FAISS similarity (k=8)
   │                                                                └─ CrossEncoderReranker (top 3)
   └─ create_stuff_documents_chain ── grounded answer + citations
```

The three mandated LangChain pillars map to:

- **Retrievers** — `src/retriever.py`: FAISS `as_retriever` wrapped in a
  `ContextualCompressionRetriever` with a `CrossEncoderReranker`.
- **Chains** — `src/rag_service.py`: `create_history_aware_retriever` +
  `create_stuff_documents_chain` composed via `create_retrieval_chain` (LCEL).
- **Memory** — `src/rag_service.py`: `RunnableWithMessageHistory` backed by
  per-session `ChatMessageHistory`.

## Project layout

```
app.py                          Streamlit UI (presentation only)
src/
  config.py                     Settings (env-overridable) + logging + validation
  vector_store.py               Embeddings + FAISS index loading
  retriever.py                  Reranking retriever
  citations.py                  Citation formatting (pure, unit-tested)
  rag_service.py                Conversational RAG chain + memory (public API)
  docling_page_loader.py        Offline: page-wise Docling PDF parsing
  docling_splitter.py           Offline: heading-aware Markdown chunking
build_docling_faiss_index.py    Offline: build the FAISS index
prompts/                        System prompts (QA + question reformulation)
tests/                          pytest (unit + gated integration)
vectorstore/faiss_docling_index Prebuilt FAISS index
data/pdfs/                      Source PDFs
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # then add your GOOGLE_API_KEY
```

Embedding and reranker models download automatically from the Hugging Face Hub
on first run (or point `EMBEDDING_MODEL` / `RERANKER_MODEL` at local copies).

## Run

```bash
streamlit run app.py
```

## Rebuild the index (optional)

Only needed if you change the PDFs or chunking. Requires Docling:

```bash
pip install docling
python build_docling_faiss_index.py
```

## Tests

```bash
pytest                                  # unit tests (no API needed)
pytest tests/test_rag_integration.py    # integration (needs GOOGLE_API_KEY + index)
```

## Configuration

All tunables live in `src/config.py` and are overridable via environment
variables — see `.env.example` for the full list (retrieval depth, rerank
top-k, temperature, history window, log level, model paths).
