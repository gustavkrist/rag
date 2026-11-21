import os

import pandas as pd
from tqdm import tqdm

from rag.documents import DocumentStore
from rag.llm import RagLLM

FILE_DIR = os.path.dirname(__file__)


def generate_and_save_answers(
    llm: RagLLM, queries: list[str], filename: str, rag: bool
) -> None:
    sample_generations = {
        "query": [],
        "context": [],
        "reply": [],
    }
    for query in tqdm(queries):
        messages, _documents = llm._get_context(query, use_rag=rag)
        if messages:
            reply, inputs = llm._generate_reply(messages, return_inputs=True)
            context = llm.tokenizer.decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            reply = reply[reply.find("</think>") + len("</think>") :].strip()
        else:
            reply = "I do not know how to respond to your question."
            context = ""
        if isinstance(context, list):
            context = context[0]
        sample_generations["query"].append(query)
        sample_generations["context"].append(context)
        sample_generations["reply"].append(reply)

    pd.DataFrame(sample_generations).to_csv(
        filename,
        index=False,
    )


def main() -> None:
    model = "qwen3-1.7B"
    data_dir = os.path.join(os.path.dirname(FILE_DIR), "data")
    db = DocumentStore(os.path.join(data_dir, "articles.json"))
    llm = RagLLM(db)
    with open(os.path.join(FILE_DIR, "sample-questions.txt")) as f:
        queries = [line.strip() for line in f]
    generate_and_save_answers(
        llm, queries, os.path.join(FILE_DIR, f"sample-generations-{model}.csv"), True
    )
    generate_and_save_answers(
        llm,
        queries,
        os.path.join(FILE_DIR, f"sample-generations-no-rag-{model}.csv"),
        False,
    )


if __name__ == "__main__":
    main()
