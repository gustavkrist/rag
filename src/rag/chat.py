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

    def generate_reply(
        self,
        query: str,
        use_rag: bool = True,
        n_documents: int = 20,
        max_distance: float = 1.05,
        enable_thinking: bool = True,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
    ) -> str:
        if use_rag:
            relevant_documents = documents.get_relevant_documents(
                query, n_results=n_documents, max_distance=max_distance
            )
            if not relevant_documents:
                return "I do not know how to respond to your question."
            document_context = "\n\n".join(
                [f"Document: {doc['content']}" for doc in relevant_documents]
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
            enable_thinking=enable_thinking,
            return_tensors="pt",
            return_dict=True,
            truncate=True,
        ).to(self.model.device)

        # Best practices for thinking/non-thinking by default https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
        generation_config = {
            "max_new_tokens": max_new_tokens or (1024 * 6 if enable_thinking else 1024),
            "temperature": temperature or (0.6 if enable_thinking else 0.7),
            "top_p": top_p or (0.95 if enable_thinking else 0.8),
            "top_k": top_k or 20,
            "min_p": min_p or 0,
        }
        outputs = self.model.generate(**inputs, **generation_config)
        reply = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        return reply
