from src.pdf_loader import load_all_pdfs
from src.text_splitter import split_documents

documents = load_all_pdfs()
chunks = split_documents(documents)

print(f"Total page documents loaded: {len(documents)}")
print(f"Total chunks created: {len(chunks)}")

for chunk in chunks[:5]:
    print("=" * 80)
    print(chunk.metadata)
    print(chunk.page_content[:500])