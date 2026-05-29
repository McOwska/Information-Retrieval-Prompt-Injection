import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text):
    """
    Very simple tokenizer for BM25.
    Lowercases text and keeps only word-like tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def load_jsonl(path):
    path = Path(path)

    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    return documents


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents

        self.tokenized_documents = [
            tokenize(self._document_to_searchable_text(doc))
            for doc in documents
        ]

        self.index = BM25Okapi(self.tokenized_documents)

    def _document_to_searchable_text(self, doc):
        """
        Combines title and text, so BM25 can match both.
        """
        title = doc.get("title", "")
        text = doc.get("text", "")

        return f"{title} {text}"

    def retrieve(self, query, top_k: int = 5):
        tokenized_query = tokenize(query)

        scores = self.index.get_scores(tokenized_query)

        ranked_results = sorted(
            zip(self.documents, scores),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for doc, score in ranked_results[:top_k]:
            results.append({
                **doc,
                "score": float(score)
            })

        return results

def main():
    corpus = load_jsonl("data/processed/corpus.jsonl")

    retriever = BM25Retriever(corpus)

    query = "film directed by Stanley Kubrick"
    results = retriever.retrieve(query, top_k=5)

    print(f"Query: {query}\n")

    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc['title']}")
        print(f"Score: {doc['score']:.4f}")
        print(doc["text"][:500])
        print("-" * 80)


if __name__ == "__main__":
    main()