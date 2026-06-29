from src.rag_chain import answer_question


def main():
    questions = [
        "What are the indicators of unaffordability of healthy diets?",
        "What are the entry points to manage agrifood systems risk and uncertainty?",
        "What desired outcomes decrease hidden costs according to the 2024 report?",
        "How do regulatory and incentive-based policies differ for land degradation?"
    ]

    for question in questions:
        result = answer_question(question)

        print("=" * 100)
        print("Question:")
        print(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation}")


if __name__ == "__main__":
    main()