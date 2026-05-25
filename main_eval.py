from retireval.retireve_document import BM25Retriever, load_jsonl
from evaluation.evaluator import run_evaluation_pipeline

def main():
    print("Loading corpus...")
    corpus = load_jsonl("data/processed/corpus.jsonl")
    
    print("Initializing BM25 Retriever...")
    retriever = BM25Retriever(corpus)
    
    run_evaluation_pipeline(retriever, limit=5)

if __name__ == "__main__":
    main()