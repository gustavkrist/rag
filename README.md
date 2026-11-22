# RAG Model Project for CP5403

## Prerequisites for running the project

> [!note]
> It is assumed that all commands are run from the repository root unless stated otherwise

- Install [`uv`](https://docs.astral.sh/uv/)
- Run `uv sync`
- Activate the venv - `source .venv/bin/activate` on Linux/MacOS

> [!note]
> The first run of the app or evaluation suite will take a while as the data is processed and the vectorized database has to be built

## Running the Web App

- With the venv active, simply run the command `rag` - the model will be loaded and the URL for the web app will be printed in the terminal
- Open the URL in a browser and chat with the model

## Running evaluation

- For the model-assisted evaluation, set the following environment variables `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` to the api key, base url, and model name for an OpenAI-compatible provider
- Run [`eval.sh`](./eval.sh) to run all the evaluation
  - One of the evaluation scripts is for human evaluation
    - This script will print a model reply to the terminal and ask for a 1-5 correctness score
    - If you want to quit this early, type `q`/`quit`/`exit` and the rest of the queries will receive NaN as their score
- Bear in mind this will likely take a while to run unless it is running on a fast GPU

### Evaluation results

- The outputs of the evaluation scripts are:
  - [`./eval/sample-generations-qwen3-1.7B.csv`](./eval/sample-generations-qwen3-1.7B.csv)
    - Contains user query, provided context, and generated reply from the RAG model for each sample question in [`./eval/sample-questions.txt`](./eval/sample-questions.txt)
  - [`./eval/sample-generations-no-rag-qwen3-1.7B.csv`](./eval/sample-generations-no-rag-qwen3-1.7B.csv)
    - Contains user query, provided context, and generated reply for the base model without RAG for each sample question in [`./eval/sample-questions.txt`](./eval/sample-questions.txt)
  - [`./eval/human-evaluation-qwen3-1.7B.csv`](./eval/human-evaluation-qwen3-1.7B.csv)
    - Contains query, context, reply, and correctness scores for the RAG model for each sample question
  - [`./eval/judge-evaluation-qwen3-1.7B.csv`](./eval/judge-evaluation-qwen3-1.7B.csv)
    - Contains query, context, reply, and:
      - Fluency score
      - Fluency justification
      - Perceived utility score
      - Perceived utility justification
      - Citation recall score
      - Citation recall justification
      - Citation precision score
      - Citation precision justification

      for the RAG model for each sample question from the LLM judge
  - [`./eval/eval-statistics-table-qwen3-1.7B.csv`](./eval/eval-statistics-table-qwen3-1.7B.csv)
    - Contains summary statistics for the human and model-assisted scores
  - [`./eval/eval-comparison-table-qwen3-1.7B.csv`](./eval/eval-comparison-table-qwen3-1.7B.csv)
- And all the same files for the base model without rag, with `-no-rag` in the name

# Legal Notices

The data found in [`data/articles.json`](./data/articles.json) is created by the official Terraria Wiki ([https://terraria.wiki.gg/](https://terraria.wiki.gg/)) and licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0).

Terraria content and materials are the intellectual property of [Re-Logic](https://re-logic.com/).
