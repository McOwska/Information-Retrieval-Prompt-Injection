from pathlib import Path

import pandas as pd

from evaluation.metrics import calculate_metrics_for_predictions


RESULTS_DIR = Path("results")
INPUT_PATH = RESULTS_DIR / "merged_results.csv"
OUTPUT_PATH = RESULTS_DIR / "evaluation_metrics.csv"

GROUND_TRUTH_COL = "Ground Truth"


def get_prediction_columns(df: pd.DataFrame) -> list[str]:
    """
    Selects columns that contain model predictions.
    Excludes question, ground truth, and metadata columns.
    """
    excluded_cols = {
        "question",
        GROUND_TRUTH_COL,
        "id",
        "type",
        "level",
        "supporting_titles",
        "poisoned_titles",
    }

    return [col for col in df.columns if col not in excluded_cols]


def main():
    df = pd.read_csv(INPUT_PATH)

    if GROUND_TRUTH_COL not in df.columns:
        raise ValueError(f"Missing ground truth column: {GROUND_TRUTH_COL}")

    prediction_cols = get_prediction_columns(df)

    if not prediction_cols:
        print("No prediction columns found.")
        return

    metrics = []

    for prediction_col in prediction_cols:
        column_metrics = calculate_metrics_for_predictions(
            predictions=df[prediction_col],
            ground_truths=df[GROUND_TRUTH_COL],
            prediction_col_name=prediction_col,
        )

        metrics.append(column_metrics)

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved metrics to: {OUTPUT_PATH}")
    print(metrics_df)


if __name__ == "__main__":
    main()