from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "merged_comparison.csv"


def main():
    rows = None

    evaluation_dirs = sorted(
        path for path in RESULTS_DIR.iterdir()
        if path.is_dir() and path.name.startswith("evaluation")
    )

    for eval_dir in evaluation_dirs:
        comparison_path = eval_dir / "comparison.csv"

        if not comparison_path.exists():
            print(f"Skipping {eval_dir.name}: no comparison.csv")
            continue

        df = pd.read_csv(comparison_path)

        if df.shape[1] < 3:
            print(f"Skipping {eval_dir.name}: comparison.csv has less than 3 columns")
            continue

        first_two_cols = df.iloc[:, :2]
        third_col = df.iloc[:, 2]

        if rows is None:
            rows = first_two_cols.copy()
        else:
            if not rows.iloc[:, :2].equals(first_two_cols):
                print(f"Warning: first two columns differ in {eval_dir.name}")

        rows[eval_dir.name] = third_col

    if rows is None:
        print("No comparison.csv files found.")
        return

    rows.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved merged file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()