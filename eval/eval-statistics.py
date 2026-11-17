import os

import pandas as pd
from rich.console import Console
from rich.markdown import Markdown
from tabulate import tabulate

FILE_DIR = os.path.dirname(__file__)


def main() -> None:
    pillars = [
        "correctness",
        "fluency",
        "perceived_utility",
        "citation_recall",
        "citation_precision",
    ]
    judge_df = pd.read_csv(os.path.join(FILE_DIR, "judge-evaluation.csv"))
    human_df = pd.read_csv(os.path.join(FILE_DIR, "human-evaluation.csv"))
    judge_df["correctness_score"] = human_df["correctness_score"]

    cols = {p.replace("_", " ").title(): judge_df[f"{p}_score"] for p in pillars}
    headers = ["Mean", "Median", "Min", "25%", "50%", "75%", "max"]
    summary_statistics = [
        [title]
        + [
            x.round(2)
            for x in (
                col.mean(),
                col.median(),
                col.min(),
                col.quantile(0.25),
                col.quantile(0.50),
                col.quantile(0.75),
                col.max(),
            )
        ]
        for title, col in cols.items()
    ]
    table = tabulate(summary_statistics, headers=headers, tablefmt="github")
    Console().print(Markdown(table))
    with open(os.path.join(FILE_DIR, "eval-statistics-table.tex"), "w") as f:
        f.write(tabulate(summary_statistics, headers=headers, tablefmt="latex"))


if __name__ == "__main__":
    main()
