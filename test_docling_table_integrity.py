from src.docling_page_loader import load_single_pdf_pagewise_docling
from src.docling_splitter import split_docling_documents


PDF_PATH = "data/pdfs/2021_pdf.pdf"


def main():
    print("Loading PDF with Docling...")
    docs = load_single_pdf_pagewise_docling(PDF_PATH)

    print(f"Total Docling page documents: {len(docs)}")

    print("\nSplitting Docling documents...")
    chunks = split_docling_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    table_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.get("content_type") == "docling_table_markdown"
    ]

    print(f"\nTotal table-like chunks found: {len(table_chunks)}")

    for i, chunk in enumerate(table_chunks, start=1):
        print("=" * 120)
        print(f"Table Chunk {i}")
        print("Metadata:")
        print(chunk.metadata)

        print("\nContent preview:")
        print(chunk.page_content[:3000])

        print("\nTable structure check:")
        lines = chunk.page_content.splitlines()
        pipe_lines = [line for line in lines if "|" in line]

        print(f"Total lines: {len(lines)}")
        print(f"Lines containing '|': {len(pipe_lines)}")

        if len(pipe_lines) >= 2:
            print("Status: Looks like Markdown table structure is preserved.")
        else:
            print("Status: Table structure may not be preserved as Markdown.")


if __name__ == "__main__":
    main()