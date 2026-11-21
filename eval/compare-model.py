JUDGE_PROMPT = """\
## Instructions
You are evaluating two question-answering systems.

## Question
Query: {query}

## System 1 reply
{reply_rag}

---

## System 2 reply
{reply_no_rag}
"""
