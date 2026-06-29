from src.docling_page_loader import load_all_pdfs_pagewise_docling
from src.docling_splitter import split_docling_documents
from src.vector_store import get_embeddings
from src.config import DOCLING_VECTORSTORE_DIR

from langchain_community.vectorstores import FAISS
import os


def main():
    print("Loading all PDFs using page-wise Docling...")
    docling_documents = load_all_pdfs_pagewise_docling()

    print(f"\nTotal Docling page documents: {len(docling_documents)}")

    print("\nSplitting Docling documents into enterprise-style chunks...")
    docling_chunks = split_docling_documents(docling_documents)

    print(f"Total Docling chunks created: {len(docling_chunks)}")

    print("\nCreating FAISS index for Docling chunks...")

    embeddings = get_embeddings()

    os.makedirs(DOCLING_VECTORSTORE_DIR, exist_ok=True)

    vectorstore = FAISS.from_documents(
        documents=docling_chunks,
        embedding=embeddings
    )

    vectorstore.save_local(DOCLING_VECTORSTORE_DIR)

    print(f"\nDocling FAISS index saved at: {DOCLING_VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()