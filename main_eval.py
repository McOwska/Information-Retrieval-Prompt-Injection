from retireval.retireve_document import BM25Retriever, load_jsonl
from evaluation.evaluator import run_evaluation_pipeline, save_results

def main():
    print("Loading corpus...")
    corpus = load_jsonl("data/processed/corpus.jsonl")
    
    print("Initializing BM25 Retriever...")
    retriever = BM25Retriever(corpus)
    
    print("Loading poisoned corpus...")
    poisoned_corpus = load_jsonl("data/processed/corpus_poisoned.jsonl")
    
    print("Initializing BM25 Retriever...")
    poisoned_retriever = BM25Retriever(poisoned_corpus)
    
    res = run_evaluation_pipeline(poisoned_retriever, limit=20, results_path="results/evaluation_baseline_poisoned")
    save_results(res, "asr, f1", "results/evaluation_baseline_poisoned/metrics.txt")


if __name__ == "__main__":
    main()