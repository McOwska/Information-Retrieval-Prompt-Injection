import argparse

from retireval.retireve_document import BM25Retriever, load_jsonl
from evaluation.evaluator import run_evaluation_pipeline, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-hop RAG evaluation.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--baseline",
        action="store_true",
        help="Run without poisoned retriever (clean corpus only).",
    )
    mode.add_argument(
        "--hop",
        type=int,
        metavar="N",
        help="Switch to poisoned retriever at hop N (1-indexed).",
    )
    parser.add_argument(
        "--results-path",
        required=True,
        help="Directory for comparison.csv and metrics.txt (e.g. results/evaluation_poisoned_embedded_1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading corpus...")
    corpus = load_jsonl("data/processed/corpus.jsonl")
    retriever = BM25Retriever(corpus)

    poisoned_hops = None
    poisoned_retriever = None

    if args.hop is not None:
        print("Loading poisoned corpus...")
        poisoned_corpus = load_jsonl("data/processed/corpus_poisoned_all_embedded.jsonl")
        poisoned_retriever = BM25Retriever(poisoned_corpus)
        poisoned_hops = [args.hop]
        print(f"Poisoned retriever active at hop(s): {poisoned_hops}")
    else:
        print("Baseline mode: no poisoned retriever.")

    print(f"Results path: {args.results_path}")
    res = run_evaluation_pipeline(
        retriever,
        limit=50,
        results_path=args.results_path,
        poisoned_hops=poisoned_hops,
        poisoned_retriever=poisoned_retriever,
    )
    save_results(res, "asr, f1", f"{args.results_path}/metrics.txt")


if __name__ == "__main__":
    main()
