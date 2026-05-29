from retrieval.retrieve_document import BM25Retriever, load_jsonl
from multihop_agent.agent import run_multihop_agent


def print_hop_logs(hops):
    for hop in hops:
        print("=" * 80)
        print(f"HOP {hop['hop']}")
        print(f"Query: {hop['query']}")
        print()

        print("Retrieved documents:")
        for i, doc in enumerate(hop["retrieved_docs"], start=1):
            print(f"{i}. {doc['title']} | score={doc['score']:.4f} | {doc['is_poisoned']}")

        print()

        print("New documents added to context:")
        for i, doc in enumerate(hop["new_docs_added"], start=1):
            print(f"{i}. {doc['title']}")


def main():
    print('okok')
    corpus = load_jsonl("data/processed/corpus_poisoned_1_hop.jsonl")
    retriever = BM25Retriever(corpus)

    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    result = run_multihop_agent(
        question=question,
        retrievers=[retriever],
        max_hops=3,
        top_k=3,
    )

    print_hop_logs(result["hops"])

    print("=" * 80)
    print("FINAL ANSWER:")
    print(result["answer"])


if __name__ == "__main__":
    main()