from src.docling_page_loader import load_single_pdf_pagewise_docling


PDF_PATH = "data/pdfs/2023_pdf.pdf"


def main():
    docs = load_single_pdf_pagewise_docling(PDF_PATH)

    print("\nDocling page-wise loading completed.")
    print(f"Total documents created: {len(docs)}")

    for i, doc in enumerate(docs[:5], start=1):
        print("=" * 100)
        print(f"Document {i}")
        print("Metadata:")
        print(doc.metadata)
        print("\nContent preview:")
        print(doc.page_content[:1500])


if __name__ == "__main__":
    main()