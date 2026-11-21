import math
import os

import pandas as pd
from rich.console import Console
from rich.markdown import Markdown

FILE_DIR = os.path.dirname(__file__)


def get_correctness_scores(samples: pd.DataFrame) -> list[float]:
    correctness_scores = []
    done = False
    for reply in samples["reply"]:
        if reply == "I do not know how to respond to your question.":
            correctness_scores.append(math.nan)
            continue
        if done:
            break
        print("\n\n\n\n\n\n\n\n\n\n")
        print("\033[H\033[2J", end="", flush=True)
        Console().print(Markdown(f"---\n\n# Reply\n\n{reply}"))
        print()
        low = 1
        high = 5
        score = math.nan
        while True:
            try:
                inp = input("Correctness Score: ")
            except EOFError:
                done = True
                break
            if inp in {"q", "quit", "exit"}:
                done = True
                break
            if inp in {"na", "nan", "NA"}:
                score = math.nan
                break
            try:
                score = int(inp)
            except ValueError:
                print("Input is not a valid integer, please try again.")
                continue
            assert isinstance(score, int)
            if score < low or high < score:
                print(f"Score must be between {low} and {high} (inclusive)")
                continue
            break
        correctness_scores.append(score)
    while len(correctness_scores) < len(samples):
        correctness_scores.append(math.nan)
    return correctness_scores



def main() -> None:
    model = "qwen3-1.7B"

    # Get evaluation for RAG model answers
    samples = pd.read_csv(os.path.join(FILE_DIR, f"sample-generations-{model}.csv"))
    df = samples.copy()
    df["correctness_score"] = get_correctness_scores(samples)
    df.to_csv(os.path.join(FILE_DIR, f"human-evaluation-{model}.csv"), index=False)

    # Get evaluation for base model answers
    samples = pd.read_csv(os.path.join(FILE_DIR, f"sample-generations-no-rag-{model}.csv"))
    df = samples.copy()
    df["correctness_score"] = get_correctness_scores(samples)
    df.to_csv(os.path.join(FILE_DIR, f"human-evaluation-no-rag-{model}.csv"), index=False)


if __name__ == "__main__":
    main()
