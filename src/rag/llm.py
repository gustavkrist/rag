from typing import Literal, TypedDict, Unpack, cast, overload

from torch.fft import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    Qwen2Tokenizer,
)
from transformers._typing import GenerativePreTrainedModel

from rag.documents import DocumentStore
from rag.types import Document, Message

# Inspired from https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-prompt-engineering#naive-rag
rag_prompt = """\
## Role
You are an assistant with expert knowledge for the video game Terraria.

## Instructions
- Answer the question by using only the information in the context below.
- If the context does not contain enough information to fully answer the question, state what you can answer and explicitely note what information is missing.
- Cite the source for each claim by using the format [Source N].
- Don't speculate or include information that isn't in the provided context.
- If none of the provided context is relevant to the question, respond with: "I don't have enough information in the available documents to answer this question."
- If the provided sources contain conflicting information, present both perspectives and cite the respective sources. Don't silently favor one source over another.

## Context
{context}

## Question
{user_query}"""

non_rag_prompt = """\
## Role
You are an expert assistant for the video game Terraria.

## Instructions
- Give a helpful answer to the user query.
- Give a clear answer in 3 to 5 sentences.

## Question
{user_query}"""


class GenerationKwargs(TypedDict, total=False):
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float


class RagLLM:
    def __init__(self, db: DocumentStore, model_name: str = "Qwen/Qwen3-1.7B") -> None:
        self.tokenizer: Qwen2Tokenizer = cast(
            Qwen2Tokenizer, AutoTokenizer.from_pretrained(model_name)
        )
        self.model: GenerativePreTrainedModel = cast(
            GenerativePreTrainedModel,
            AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
        )
        self.documents = db

    @overload
    def _generate_reply(
        self,
        messages: list[Message],
        enable_thinking: bool = True,
        *,
        return_inputs: Literal[False] = False,
        **kwargs: Unpack[GenerationKwargs],
    ) -> str: ...

    @overload
    def _generate_reply(
        self,
        messages: list[Message],
        enable_thinking: bool = True,
        *,
        return_inputs: Literal[True],
        **kwargs: Unpack[GenerationKwargs],
    ) -> tuple[str, BatchEncoding[Tensor]]: ...

    @overload
    def _generate_reply(
        self,
        messages: list[Message],
        enable_thinking: bool = True,
        *,
        return_inputs: bool,
        **kwargs: Unpack[GenerationKwargs],
    ) -> str | tuple[str, BatchEncoding[Tensor]]: ...

    def _generate_reply(
        self,
        messages: list[Message],
        enable_thinking: bool = True,
        *,
        return_inputs: bool = False,
        **kwargs: Unpack[GenerationKwargs],
    ) -> str | tuple[str, BatchEncoding[Tensor]]:
        inputs = self.tokenizer.apply_chat_template(
            cast(list[dict[str, str]], messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
            return_dict=True,
            truncate=True,
        )
        assert isinstance(inputs, BatchEncoding)
        inputs = inputs.to(self.model.device)

        # Best practices for thinking/non-thinking by default https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 32_768),
            "temperature": kwargs.get("temperature", (0.6 if enable_thinking else 0.7)),
            "top_p": kwargs.get("top_p", (0.95 if enable_thinking else 0.8)),
            "top_k": kwargs.get("top_k", 20),
            "min_p": kwargs.get("min_p", 0.0),
        }

        outputs = self.model.generate(**inputs, **generation_config)
        reply = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        assert isinstance(reply, str)

        if return_inputs:
            return reply, inputs
        else:
            return reply

    def _get_context(
        self,
        query: str,
        use_rag: bool = True,
        rerank: bool = True,
        n_results: int = 15,
        n_documents: int = 5,
        max_distance: float = 2,
    ) -> tuple[list[Message], list[Document]]:
        documents = []
        if use_rag:
            if rerank:
                documents = self.documents.get_relevant_documents(
                    query,
                    n_results=n_results,
                    top_k=n_documents,
                    max_distance=max_distance,
                )
            else:
                documents = self.documents.retrieve_documents(
                    query, n_results=n_documents, max_distance=max_distance
                )
            if not documents:
                return [], []
            document_context = "\n\n---\n\n".join(
                [
                    f"[Source {i}]: {doc['content']}"
                    for i, doc in enumerate(documents, start=1)
                ]
            )
            model_context = rag_prompt.format(
                context=document_context, user_query=query
            )
        else:
            model_context = non_rag_prompt.format(user_query=query)

        messages: list[Message] = [
            {"role": "system", "content": model_context},
        ]
        return messages, documents

    def generate_reply(
        self,
        query: str,
        use_rag: bool = True,
        rerank: bool = True,
        n_results: int = 15,
        n_documents: int = 5,
        max_distance: float = 2,
        enable_thinking: bool = True,
        **generation_kwargs: Unpack[GenerationKwargs],
    ) -> tuple[str, list[Document]]:
        messages, documents = self._get_context(
            query=query,
            use_rag=use_rag,
            rerank=rerank,
            n_results=n_results,
            n_documents=n_documents,
            max_distance=max_distance,
        )
        if not messages:
            return "I do not know how to respond to your question.", []

        reply = self._generate_reply(messages, enable_thinking, **generation_kwargs)

        if enable_thinking:
            reply = reply[reply.find("</think>") + len("</think>") :].strip()

        return reply, documents
