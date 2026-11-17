import math
import os
from typing import Annotated

import openai
import pandas as pd
from pydantic import BaseModel, Field
from tqdm.contrib.concurrent import thread_map

# Inspired by https://github.com/trevorandersen/CP5403-pracs/blob/414dfbd74c86550b5292b967ca17d3086490ddcb/week08/cp5403-prac-08.ipynb
JUDGE_PROMPT = """You are evaluating a question-answering system.

Question: {query}

Context provided to the system:
{context}

System's answer: {reply}

Score the answer on the following criteria on a scale of 1-5:
Fluency: Is the generated text fluent and cohesive?
Perceived utility: Is the answer helpful and informative?
Citation recall: Proportion of generated statements about the external world that are fully supported by their citations
Citation precision: Proportion of generated citations that support their associated statements

Reply with the score and one sentence of justification for each"""


class JudgeResponse(BaseModel):
    fluency_score: Annotated[int, Field(ge=1, le=5)]
    fluency_justification: str
    perceived_utility_score: Annotated[int, Field(ge=1, le=5)]
    perceived_utility_justification: str
    citation_recall_score: Annotated[int, Field(ge=1, le=5)]
    citation_recall_justification: str
    citation_precision_score: Annotated[int, Field(ge=1, le=5)]
    citation_precision_justification: str


def judge_response(response_tuple: tuple[str, str, str]) -> JudgeResponse | None:
    query, context, reply = response_tuple
    prompt = JUDGE_PROMPT.format(query=query, context=context, reply=reply)
    client = openai.Client(
        api_key=os.environ["NEURALWATT_API_KEY"],
        base_url="https://api.neuralwatt.com/v1",
    )
    response = client.responses.parse(
        model="deepseek-v4.1-flash", input=prompt, text_format=JudgeResponse
    )
    return response.output_parsed


def main() -> None:
    samples = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "sample-generations.csv")
    )
    pillars = ["fluency", "perceived_utility", "citation_recall", "citation_precision"]
    judge_scores = {p: [] for p in pillars}
    judge_justifications = {p: [] for p in pillars}
    rows = [(q, c, r) for _, (q, c, r) in samples.iterrows()]
    judgings: list[JudgeResponse | None] = thread_map(
        judge_response, rows, max_workers=3
    )
    for j in judgings:
        for p in pillars:
            judge_scores[p].append(getattr(j, f"{p}_score", math.nan))
            judge_justifications[p].append(getattr(j, f"{p}_justification", ""))
    df = samples.copy()
    for p in pillars:
        df[f"judge_{p}_score"] = judge_scores[p]
        df[f"judge_{p}_justification"] = judge_justifications[p]
    df.to_csv(
        os.path.join(os.path.dirname(__file__), "judge-evaluation.csv"), index=False
    )


if __name__ == "__main__":
    main()
