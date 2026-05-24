import json
from pathlib import Path

from datasets import load_dataset

from poisoning import poison_document


CLEAN_CORPUS_PATH = Path("data/processed/corpus.jsonl")
POISONED_CORPUS_PATH = Path("data/processed/corpus_poisoned.jsonl")
QUESTIONS_PATH = Path("data/processed/questions.jsonl")

NUM_EXAMPLES = 100

# Injection style: 'explicit', 'embedded', or 'misleading'.
# 'embedded' and 'misleading' require a live LLM call per poisoned document.
POISON_STYLE = "explicit"

# Which supporting doc to poison per question: 1 = first supporting title, 2 = second.
TARGET_HOP = 1


def make_doc_id(title: str) -> str:
    return title.replace(" ", "_").replace("/", "_")


def save_jsonl(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_corpus_and_questions(train_data) -> tuple[dict, list]:
    """
    Builds a clean corpus dict keyed by title and a list of question dicts.
    supporting_titles preserves insertion order (first occurrence = hop 1 doc).
    Each question records poisoned_title: the doc that will be targeted.
    """
    corpus_by_title: dict[str, dict] = {}
    questions: list[dict] = []

    for example in train_data:
        example_id = example["id"]

        titles = example["context"]["title"]
        sentences_list = example["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences)
            if title not in corpus_by_title:
                corpus_by_title[title] = {
                    "doc_id": make_doc_id(title),
                    "title": title,
                    "text": text,
                    "is_poisoned": False,
                    "source_example_ids": [example_id],
                }
            else:
                corpus_by_title[title]["source_example_ids"].append(example_id)

        # Preserve order so index 0 = first supporting doc (hop 1), index 1 = second (hop 2)
        seen = set()
        supporting_titles = []
        for t in example["supporting_facts"]["title"]:
            if t not in seen:
                seen.add(t)
                supporting_titles.append(t)

        target_index = TARGET_HOP - 1
        poisoned_title = (
            supporting_titles[target_index]
            if target_index < len(supporting_titles)
            else supporting_titles[0]
        )

        questions.append({
            "id": example_id,
            "question": example["question"],
            "answer": example["answer"],
            "type": example["type"],
            "level": example["level"],
            "supporting_titles": supporting_titles,
            "poisoned_title": poisoned_title,
            "supporting_facts": [
                {"title": title, "sent_id": sent_id}
                for title, sent_id in zip(
                    example["supporting_facts"]["title"],
                    example["supporting_facts"]["sent_id"],
                )
            ],
        })

    return corpus_by_title, questions


def build_poisoned_corpus(
    corpus_by_title: dict,
    questions: list,
    style: str,
    target_hop: int,
    call_llm=None,
) -> list[dict]:
    """
    Returns a corpus where exactly one doc per question is poisoned.
    If multiple questions share the same target doc, the first question wins
    and subsequent ones reuse the already-poisoned entry.
    """
    title_to_target: dict[str, tuple[str, int]] = {}
    for q in questions:
        title = q["poisoned_title"]
        if title not in title_to_target:
            title_to_target[title] = (q["id"], target_hop)

    poisoned_corpus = []
    for title, doc in corpus_by_title.items():
        if title in title_to_target:
            question_id, hop = title_to_target[title]
            poisoned_corpus.append(
                poison_document(
                    doc=doc,
                    style=style,
                    target_question_id=question_id,
                    target_hop=hop,
                    call_llm=call_llm,
                )
            )
        else:
            poisoned_corpus.append({**doc})

    return poisoned_corpus


def main():
    call_llm = None
    if POISON_STYLE in ("embedded", "misleading"):
        from llm import call_llm as _call_llm  # noqa: PLC0415
        call_llm = _call_llm

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
    train_data = dataset["train"].select(range(NUM_EXAMPLES))

    corpus_by_title, questions = build_corpus_and_questions(train_data)

    clean_corpus = list(corpus_by_title.values())
    save_jsonl(clean_corpus, CLEAN_CORPUS_PATH)
    print(f"Saved {len(clean_corpus)} documents to {CLEAN_CORPUS_PATH}")

    poisoned_corpus = build_poisoned_corpus(
        corpus_by_title=corpus_by_title,
        questions=questions,
        style=POISON_STYLE,
        target_hop=TARGET_HOP,
        call_llm=call_llm,
    )
    poisoned_count = sum(1 for d in poisoned_corpus if d.get("is_poisoned"))
    save_jsonl(poisoned_corpus, POISONED_CORPUS_PATH)
    print(f"Saved {len(poisoned_corpus)} documents ({poisoned_count} poisoned) to {POISONED_CORPUS_PATH}")

    save_jsonl(questions, QUESTIONS_PATH)
    print(f"Saved {len(questions)} questions to {QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
