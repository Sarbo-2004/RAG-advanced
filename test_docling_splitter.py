from src.docling_page_loader import load_single_pdf_pagewise_docling
from src.docling_splitter import split_docling_documents


PDF_PATH = "data/pdfs/2023_pdf.pdf"


def main():
    print("Loading PDF with page-wise Docling...")
    docs = load_single_pdf_pagewise_docling(PDF_PATH)

    print(f"\nTotal Docling page documents: {len(docs)}")

    print("\nSplitting Docling documents...")
    chunks = split_docling_documents(docs)

    print(f"Total Docling chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:8], start=1):
        print("=" * 100)
        print(f"Chunk {i}")
        print("Metadata:")
        print(chunk.metadata)
        print("\nContent preview:")
        print(chunk.page_content[:1000])


if __name__ == "__main__":
    main()