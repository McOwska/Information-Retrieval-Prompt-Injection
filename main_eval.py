from retireval.retireve_document import BM25Retriever, load_jsonl
from evaluation.evaluator import run_evaluation_pipeline, save_results

def main():    
    print("Loading poisoned corpora...")
    poisoned_corpus_1 = load_jsonl("data/processed/corpus_poisoned_1_hop.jsonl")
    poisoned_corpus_2 = load_jsonl("data/processed/corpus_poisoned_2_hop.jsonl")
    poisoned_corpus_3 = load_jsonl("data/processed/corpus_poisoned_3_hop.jsonl")
    
    print("Initializing Poisoned Retriever...")
    retriever_1 = BM25Retriever(poisoned_corpus_1)
    retriever_2 = BM25Retriever(poisoned_corpus_2)
    retriever_3 = BM25Retriever(poisoned_corpus_3)

    retrievers = [retriever_1, retriever_2, retriever_3]
    
    res = run_evaluation_pipeline(retrievers, limit=20, results_path="results/evaluation_poisoned_embedded_warning_3", poisoned_hops=[3], poisoned_retriever=poisoned_retriever)
    save_results(res, "asr, f1", "results/evaluation_poisoned_embedded_warning_3/metrics.txt")


if __name__ == "__main__":
    main()