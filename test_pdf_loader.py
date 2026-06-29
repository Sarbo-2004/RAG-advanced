from src.pdf_loader import load_all_pdfs

documents = load_all_pdfs()

print(f"Total page documents loaded: {len(documents)}")

for doc in documents[:3]:
    print("=" * 80)
    print(doc.metadata)
    print(doc.page_content[:500])