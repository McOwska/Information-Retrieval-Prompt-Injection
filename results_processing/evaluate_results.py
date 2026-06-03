import json
from pathlib import Path

import pandas as pd

from evaluation.metrics import calculate_metrics_for_predictions


RESULTS_DIR = Path("results")
INPUT_PATH = RESULTS_DIR / "merged_results.csv"
OUTPUT_PATH = RESULTS_DIR / "evaluation_metrics.csv"
QUESTIONS_PATH = Path("data/processed/questions.jsonl")

GROUND_TRUTH_COL = "Ground Truth"
NORMALIZED_QUESTION_COL = "normalized_question"
QUESTION_TYPES_TO_EVALUATE = ["bridge"]

EXCLUDED_COLS_LOWER = {
    "question",
    "normalized_question",
    "ground truth",
    "id",
    "type",
    "level",
    "supporting_titles",
    "poisoned_titles",
}


def normalize_question(question: str) -> str:
    return " ".join(str(question).replace(",", " ").split())


def load_ground_truths_by_normalized_question(
    filepath: Path = QUESTIONS_PATH,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data.get("question")
            if not question or data.get("type") not in QUESTION_TYPES_TO_EVALUATE:
                continue
            lookup[normalize_question(question)] = data.get("answer", "")
    return lookup


def resolve_ground_truths(df: pd.DataFrame, lookup: dict[str, str]) -> pd.Series:
    """
    Prefer answers from questions.jsonl (by normalized question).
    Falls back to the merged CSV Ground Truth column when present.
    """
    if NORMALIZED_QUESTION_COL in df.columns:
        keys = df[NORMALIZED_QUESTION_COL]
    else:
        question_col = next(
            (c for c in df.columns if c.lower() == "question"),
            None,
        )
        if question_col is None:
            raise ValueError("No question column found in merged results")
        keys = df[question_col].apply(normalize_question)

    from_jsonl = keys.map(lookup)
    if GROUND_TRUTH_COL in df.columns:
        csv_gt = df[GROUND_TRUTH_COL].replace("", pd.NA)
        return from_jsonl.fillna(csv_gt).fillna("")
    return from_jsonl.fillna("")


def get_prediction_columns(df: pd.DataFrame) -> list[str]:
    """
    Selects columns that contain model predictions.
    Excludes question, ground truth, and metadata columns.
    """
    return [
        col
        for col in df.columns
        if col.strip().lower() not in EXCLUDED_COLS_LOWER
    ]


def main():
    df = pd.read_csv(INPUT_PATH)

    lookup = load_ground_truths_by_normalized_question()
    ground_truths = resolve_ground_truths(df, lookup)

    prediction_cols = get_prediction_columns(df)

    if not prediction_cols:
        print("No prediction columns found.")
        return

    metrics = []

    for prediction_col in prediction_cols:
        column_metrics = calculate_metrics_for_predictions(
            predictions=df[prediction_col],
            ground_truths=ground_truths,
            prediction_col_name=prediction_col,
        )

        metrics.append(column_metrics)

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved metrics to: {OUTPUT_PATH}")
    print(metrics_df)


if __name__ == "__main__":
    main()