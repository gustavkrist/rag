import itertools
import json
import logging
import os
import re
from collections import defaultdict
from typing import cast

import chromadb
from sentence_transformers import CrossEncoder

from rag.types import Document

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.reranker = CrossEncoder(model)

    def rerank(
        self, query: str, docs: list[Document], min_score: float = 0
    ) -> list[Document]:
        pairs = [(query, doc["content"]) for doc in docs]
        scores = self.reranker.predict(pairs)
        logger.debug(
            [(s, d["title"]) for s, d in sorted(zip(scores, docs), reverse=True)]
        )
        return [
            x[1] for x in sorted(zip(scores, docs), reverse=True) if x[0] > min_score
        ]


class DocumentStore:
    def __init__(self, data_path: str = "data/articles.json") -> None:
        self.db = _get_document_db(data_path)
        self.reranker = Reranker()

    def retrieve_documents(
        self, query: str, n_results: int = 5, max_distance: float = 1.1
    ) -> list[Document]:
        result = self.db.query(query_texts=query, n_results=n_results)
        assert result["documents"] is not None
        assert result["distances"] is not None
        assert result["metadatas"] is not None
        documents: list[tuple[Document, float]] = [
            (
                {
                    "id": id,
                    "title": cast(str, metadata["title"]),
                    "content": content,
                    "url": cast(str, metadata["url"]),
                },
                dist,
            )
            for id, metadata, content, dist in zip(
                result["ids"][0],
                result["metadatas"][0],
                result["documents"][0],
                result["distances"][0],
            )
        ]
        retrieved_documents: list[Document] = [
            doc for (doc, dist) in documents if dist < max_distance
        ]
        return retrieved_documents

    def get_relevant_documents(
        self,
        query: str,
        n_results: int = 15,
        max_distance: float = 2,
        top_k: int = 5,
        min_similarity_score: float = 0,
    ) -> list[Document]:
        retrieved_documents = self.retrieve_documents(
            query=query, n_results=n_results, max_distance=max_distance
        )
        logger.debug("retrieved %s documents", len(retrieved_documents))
        reranked_articles = self.reranker.rerank(
            query, retrieved_documents, min_similarity_score
        )
        return reranked_articles[:top_k]


def _get_document_db(data_path: str) -> chromadb.Collection:
    client = chromadb.PersistentClient("./.chroma")
    if "rag" not in [c.name for c in client.list_collections()]:
        if os.path.isdir(data_path):
            docs = read_documents_from_folder(data_path)
        else:
            docs = read_documents_from_file(data_path)
        create_document_db(docs)
    return client.get_collection("rag")


def create_document_db(documents: list[Document]) -> None:
    client = chromadb.PersistentClient("./.chroma")

    doc_ids = [doc["id"] for doc in documents]
    doc_contents = [doc["content"] for doc in documents]
    doc_metadatas: list[chromadb.Metadata] = [
        {"title": doc["title"], "url": doc["url"]} for doc in documents
    ]

    collection = client.get_or_create_collection("rag")
    max_batch_size = 5461
    for ids, contents, metadatas in zip(
        *(
            map(list, itertools.batched(l, max_batch_size))
            for l in (doc_ids, doc_contents, doc_metadatas)
        )
    ):
        collection.upsert(ids=ids, documents=contents, metadatas=metadatas)


def read_documents_from_folder(path: str) -> list[Document]:
    documents: list[Document] = []
    loaded_ids: set[str] = set()
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(path + os.sep + filename, "r", encoding="UTF-8") as f:
                loaded = json.load(f)
                loaded.pop("categories")
                if loaded["id"] not in loaded_ids:
                    documents.append(cast(Document, loaded))
                    loaded_ids.add(loaded["id"])
    return documents


def read_documents_from_file(filepath: str) -> list[Document]:
    with open(filepath, "r", encoding="UTF-8") as f:
        loaded = json.load(f)
        docs: list[Document] = []
        loaded_ids: set[str] = set()
        for doc in loaded:
            doc.pop("categories")
            if doc["id"] not in loaded_ids:
                docs.append(doc)
                loaded_ids.add(doc["id"])
    return list(
        itertools.chain(*[chunk_document(doc) for doc in docs if doc["content"]])
    )


def chunk_document(doc: Document) -> list[Document]:
    """
    Documents have the form

    == <headline> ==
    content...

    Chunks documents into one chunk for each section/subsection
    """
    pattern = re.compile(r"^===?([^=]+)===?((?:(?:.|\n)(?!==))*)", flags=re.MULTILINE)
    matches: list[tuple[str, str]] = re.findall(
        pattern, "== Intro ==\n" + doc["content"]
    )
    counts = defaultdict(int)
    documents: list[Document] = []
    for subsec, content in matches:
        subsec_title = subsec.strip()
        if subsec_title in {
            "History",
            "References",
            "Shimmered form",
            "Achievements",
        } or subsec_title.startswith("Bestiary entries"):
            continue
        url = f"{doc['url']}#{subsec_title.replace(' ', '_')}"
        if counts[subsec_title] > 0:
            url += "_" + str(counts[subsec_title] + 1)

        # TODO: Chunk longer contents perhaps
        documents.append(
            {
                "id": f"{doc['id']}-{subsec_title}-{counts[subsec_title]}",
                "title": f"{doc['title']} - {subsec.strip()}",
                "content": f"{doc['title']} - {subsec.strip()}\n{content.strip()}",
                "url": url,
            }
        )

        counts[subsec_title] = counts[subsec_title] + 1

    return documents
