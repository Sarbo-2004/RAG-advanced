from src.table_loader import load_all_tables


def main():
    table_docs = load_all_tables()

    print(f"Total table documents extracted: {len(table_docs)}")

    for index, doc in enumerate(table_docs[:5], start=1):
        print("=" * 100)
        print(f"Document {index}")
        print("Metadata:", doc.metadata)
        print("Content preview:")
        print(doc.page_content[:1000])
        print()


if __name__ == "__main__":
    main()