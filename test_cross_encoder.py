from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL


def main():
    model = CrossEncoder(RERANKER_MODEL)

    query = "What are the three main ways to manage agrifood systems risk and uncertainty?"

    documents = [
        "The three main ways are ensuring diversity, managing connectivity, and managing risks.",
        "True cost accounting measures hidden costs of agrifood systems.",
        "Land degradation policies include regulatory and incentive-based measures."
    ]

    pairs = [[query, doc] for doc in documents]

    scores = model.predict(pairs)

    for doc, score in zip(documents, scores):
        print("=" * 80)
        print("Score:", score)
        print("Document:", doc)


if __name__ == "__main__":
    main()