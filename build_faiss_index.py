from src.pdf_loader import load_all_pdfs
from src.text_splitter import split_documents
from src.table_loader import load_all_tables
from src.vector_store import create_faiss_index


def main():
    print("Loading PDF text documents...")
    text_documents = load_all_pdfs()
    print(f"Total text page documents: {len(text_documents)}")

    print("\nSplitting text documents into chunks...")
    text_chunks = split_documents(text_documents)
    print(f"Total text chunks: {len(text_chunks)}")

    print("\nLoading table documents...")
    table_documents = load_all_tables()
    print(f"Total table chunks: {len(table_documents)}")

    all_chunks = text_chunks + table_documents
    print(f"\nTotal chunks going into FAISS: {len(all_chunks)}")

    print("\nCreating FAISS index...")
    create_faiss_index(all_chunks)

    print("\nFAISS index created successfully with text + table chunks.")


if __name__ == "__main__":
    main()