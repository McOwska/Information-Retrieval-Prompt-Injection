from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")
QUESTIONS_PATH = Path("data/processed/questions.jsonl")
OUTPUT_PATH = RESULTS_DIR / "merged_results.csv"


def normalize_question(question: str) -> str:
    return " ".join(str(question).replace(",", " ").split())


def load_questions_metadata():
    df = pd.read_json(QUESTIONS_PATH, lines=True)

    df["normalized_question"] = df["question"].apply(normalize_question)

    metadata_cols = [
        "normalized_question",
    ]

    return df[metadata_cols]


def main():
    rows = None

    model_dirs = sorted(
        path for path in RESULTS_DIR.iterdir()
        if path.is_dir()
    )

    for model_dir in model_dirs:
        evaluation_dirs = sorted(
            path for path in model_dir.iterdir()
            if path.is_dir()
        )

        for eval_dir in evaluation_dirs:
            comparison_path = eval_dir / "comparison.csv"

            if not comparison_path.exists():
                print(f"Skipping {model_dir.name}/{eval_dir.name}: no comparison.csv")
                continue

            df = pd.read_csv(comparison_path)

            if df.shape[1] < 3:
                print(
                    f"Skipping {model_dir.name}/{eval_dir.name}: "
                    "comparison.csv has less than 3 columns"
                )
                continue

            first_two_cols = df.iloc[:, :2]
            third_col = df.iloc[:, 2]

            if rows is None:
                rows = first_two_cols.copy()

                question_col = rows.columns[0]
                rows["normalized_question"] = rows[question_col].apply(normalize_question)

            else:
                current_question_col = first_two_cols.columns[0]

                current_normalized_questions = first_two_cols[current_question_col].apply(
                    normalize_question
                )

                if not rows["normalized_question"].equals(current_normalized_questions):
                    print(
                        f"Warning: questions differ in "
                        f"{model_dir.name}/{eval_dir.name}"
                    )

            model_name = model_dir.name.replace("results_", "")
            column_name = f"{model_name}_{eval_dir.name}"

            rows[column_name] = third_col

    if rows is None:
        print("No comparison.csv files found.")
        return

    rows.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved merged file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()