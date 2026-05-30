from retireval.retireve_document import BM25Retriever, load_jsonl
from evaluation.evaluator import run_evaluation_pipeline, save_results

def main():
    print("Loading corpus...")
    corpus = load_jsonl("data/processed/corpus.jsonl")
    
    print("Initializing BM25 Retriever...")
    retriever = BM25Retriever(corpus)
    
    print("Loading poisoned corpus...")
    poisoned_corpus = load_jsonl("data/processed/corpus_poisoned_all_embedded.jsonl")
    
    print("Initializing BM25 Retriever...")
    poisoned_retriever = BM25Retriever(poisoned_corpus)
    
    res = run_evaluation_pipeline(retriever, limit=50, results_path="results/evaluation_poisoned_embedded_1", poisoned_hops=[1], poisoned_retriever=poisoned_retriever)
    save_results(res, "asr, f1", "results/evaluation_poisoned_embedded_1/metrics.txt")


if __name__ == "__main__":
    main()