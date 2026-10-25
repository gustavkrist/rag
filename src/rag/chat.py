from typing import cast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TokenizersBackend,
)

from rag import documents

rag_prompt = """You are an expert assistant for the video game Terraria.

Answer the user's question using only the retrieved assessment documents below.
Answer in a helpful way for a gamer.

Rules:
1. Use only the retrieved documents as your source of truth.
2. If the answer is not explicitly found in the documents, say: "The retrieved documents do not provide enough information."
3. Do not infer, guess, or answer from general knowledge.
4. Do not make up items, bosses, NPCs, or crafting recipies that are not stated in the documents
5. Give a clear explanation in 3 to 5 sentences only when the answer is supported by the retrieved documents.

Retrieved documents:
{context}"""

non_rag_prompt = """You are an expert assistant for the video game Terraria.
Answer in a helpful way for a gamer."""


class LLM:
    def __init__(self, model_name: str):
        self.tokenizer: TokenizersBackend = cast(
            TokenizersBackend, AutoTokenizer.from_pretrained(model_name)
        )
        self.model: PreTrainedModel = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
        )

    def generate_reply(self, query: str, use_rag: bool = True) -> str:
        if use_rag:
            relevant_documents = documents.get_relevant_documents(
                query, n_results=20, max_distance=1.05
            )
            if not relevant_documents:
                return "I do not know how to respond to your question."
            document_context = "\n\n".join(
                [
                    f"Document: {doc['content']}"
                    for doc in relevant_documents
                ]
            )
            model_context = rag_prompt.format(context=document_context)
        else:
            model_context = non_rag_prompt

        messages = [
            {"role": "system", "content": model_context},
            {"role": "user", "content": query},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_tensors="pt",
            return_dict=True,
            truncate=True,
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        reply = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        return reply
