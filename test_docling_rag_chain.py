from src.docling_rag_chain import answer_question_docling


def main():
    questions = [
        "What is true cost accounting?",
        "What are hidden costs in agrifood systems?",
        "Are hidden costs higher in low-income countries?",
        "What does the 2023 report say about the two-phase assessment process?",
        "What are the dominant quantified hidden costs globally?"
    ]

    for question in questions:
        result = answer_question_docling(question)

        print("=" * 100)
        print("Question:")
        print(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation}")

        print("\nRetrieved Sources:")
        for doc in result["source_documents"]:
            print("-" * 80)
            print(doc.metadata)
            print(doc.page_content[:500])


if __name__ == "__main__":
    main()