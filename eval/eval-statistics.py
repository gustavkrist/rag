import os

import pandas as pd

FILE_DIR = os.path.dirname(__file__)


def main() -> None:
    model = "qwen3-1.7B"
    pillars = [
        "correctness",
        "fluency",
        "perceived_utility",
        "citation_recall",
        "citation_precision",
    ]
    judge_df = pd.read_csv(os.path.join(FILE_DIR, f"judge-evaluation-{model}.csv"))
    human_df = pd.read_csv(os.path.join(FILE_DIR, f"human-evaluation-{model}.csv"))
    judge_df["correctness_score"] = human_df["correctness_score"]

    cols = {p.replace("_", " ").title(): judge_df[f"{p}_score"] for p in pillars}
    statistics = {
        "Score": [],
        "Mean": [],
        "Median": [],
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": [],
    }
    for title, col in cols.items():
        statistics["Score"].append(title)
        statistics["Mean"].append(round(col.mean(), 2))
        statistics["Median"].append(col.median())
        statistics["Min"].append(col.min())
        statistics["25%"].append(round(col.quantile(0.25), 2))
        statistics["50%"].append(round(col.quantile(0.50), 2))
        statistics["75%"].append(round(col.quantile(0.75), 2))
        statistics["Max"].append(col.max())

    pd.DataFrame(statistics).to_csv(
        os.path.join(FILE_DIR, f"eval-statistics-table-{model}.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
