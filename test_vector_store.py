from src.pdf_loader import load_all_pdfs
from src.text_splitter import split_documents
from src.vector_store import create_faiss_index, load_faiss_index

documents = load_all_pdfs()
chunks = split_documents(documents)

print(f"Total documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

vectorstore = create_faiss_index(chunks)

print("FAISS index created and saved successfully.")

loaded_vectorstore = load_faiss_index()

query = "What is true cost accounting?"
results = loaded_vectorstore.similarity_search(query, k=3)

print(f"\nQuery: {query}")
print(f"Retrieved results: {len(results)}")

for result in results:
    print("=" * 80)
    print(result.metadata)
    print(result.page_content[:500])