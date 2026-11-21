import os

import pandas as pd

FILE_DIR = os.path.dirname(__file__)


def compute_and_save_stats(df: pd.DataFrame, filename: str) -> None:
    pillars = [
        "correctness",
        "fluency",
        "perceived_utility",
        "citation_recall",
        "citation_precision",
        "preciseness",
    ]

    cols = {p.replace("_", " ").title(): df[f"{p}_score"] for p in pillars}
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
        filename,
        index=False,
    )


def compare_and_save(
    rag_df: pd.DataFrame, no_rag_df: pd.DataFrame, filename: str
) -> None:
    pillars = [
        "correctness",
        "fluency",
        "perceived_utility",
        "citation_recall",
        "citation_precision",
        "preciseness",
    ]

    rag_cols = {p.replace("_", " ").title(): rag_df[f"{p}_score"] for p in pillars}
    no_rag_cols = {
        p.replace("_", " ").title(): no_rag_df[f"{p}_score"] for p in pillars
    }

    comparison = {
        "": [title for title in rag_cols],
        "RAG": [float(round(col.mean(), 2)) for title, col in rag_cols.items()],
        "Base": [float(round(col.mean(), 2)) for title, col in no_rag_cols.items()],
    }

    pd.DataFrame(comparison).to_csv(
        filename,
        index=False,
    )


def main() -> None:
    model = "qwen3-1.7B"
    judge_df = pd.read_csv(os.path.join(FILE_DIR, f"judge-evaluation-{model}.csv"))
    human_df = pd.read_csv(os.path.join(FILE_DIR, f"human-evaluation-{model}.csv"))
    judge_df["correctness_score"] = human_df["correctness_score"]
    compute_and_save_stats(
        judge_df, os.path.join(FILE_DIR, f"eval-statistics-table-{model}.csv")
    )

    judge_df_no_rag = pd.read_csv(
        os.path.join(FILE_DIR, f"judge-evaluation-no-rag-{model}.csv")
    )
    human_df_no_rag = pd.read_csv(
        os.path.join(FILE_DIR, f"human-evaluation-no-rag-{model}.csv")
    )
    judge_df_no_rag["correctness_score"] = human_df_no_rag["correctness_score"]
    compute_and_save_stats(
        judge_df_no_rag,
        os.path.join(FILE_DIR, f"eval-statistics-table-no-rag-{model}.csv"),
    )

    compare_and_save(
        judge_df,
        judge_df_no_rag,
        os.path.join(FILE_DIR, f"eval-comparison-table-{model}.csv"),
    )


if __name__ == "__main__":
    main()
