import json
from pathlib import Path
from datasets import load_dataset

from poisoning import poison_content_with_set_string


CORPUS_PATH = Path("data/processed/corpus_poisoned.jsonl")
QUESTIONS_PATH = Path("data/processed/questions.jsonl")

POISONING = True


def make_doc_id(title):
    return title.replace(" ", "_").replace("/", "_")


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
    train_data = dataset["train"].select(range(100))

    corpus_by_title = {}
    questions = []

    for example in train_data:
        example_id = example["id"]

        titles = example["context"]["title"]
        sentences_list = example["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)

            if POISONING:
                text = poison_content_with_set_string(text)

            if title not in corpus_by_title:
                corpus_by_title[title] = {
                    "doc_id": make_doc_id(title),
                    "title": title,
                    "text": text,
                    "source_example_ids": [example_id],
                    "is_poisoned": POISONING,
                }
            else:
                corpus_by_title[title]["source_example_ids"].append(example_id)

        supporting_titles = list(set(example["supporting_facts"]["title"]))

        questions.append({
            "id": example_id,
            "question": example["question"],
            "answer": example["answer"],
            "type": example["type"],
            "level": example["level"],
            "supporting_titles": supporting_titles,
            "supporting_facts": [
                {
                    "title": title,
                    "sent_id": sent_id,
                }
                for title, sent_id in zip(
                    example["supporting_facts"]["title"],
                    example["supporting_facts"]["sent_id"],
                )
            ],
        })

    corpus = list(corpus_by_title.values())

    save_jsonl(corpus, CORPUS_PATH)
    save_jsonl(questions, QUESTIONS_PATH)

    print(f"Saved {len(corpus)} documents to {CORPUS_PATH}")
    print(f"Saved {len(questions)} questions to {QUESTIONS_PATH}")


if __name__ == "__main__":
    main()