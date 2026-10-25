import itertools
import json
import os
import re
from typing import TypedDict, cast

import chromadb


class Document(TypedDict):
    id: str
    title: str
    content: str


def read_documents_from_folder(path: str) -> list[Document]:
    documents: list[Document] = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(path + os.sep + filename, "r", encoding="UTF-8") as f:
                documents.append(cast(Document, json.load(f)))
    return documents


def read_documents_from_file(filepath: str) -> list[Document]:
    with open(filepath, "r", encoding="UTF-8") as f:
        docs = cast(list[Document], json.load(f))
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
    pattern = re.compile(r"^===*([^=]+)===*((?:(?:.|\n)(?!==))*)", flags=re.MULTILINE)
    matches: list[tuple[str, str]] = re.findall(
        pattern, "== Intro ==\n" + doc["content"]
    )
    documents: list[Document] = [
        {
            "id": f"{doc['id']}-{subsec.strip()}-{i}",
            "title": f"{doc['title']} - {subsec.strip()}",
            "content": f"{doc['title']} - {subsec.strip()}\n{content.strip()}",
        }
        for i, (subsec, content) in enumerate(matches)
        if content.strip()
    ]
    return documents


def create_document_db(documents: list[Document]) -> None:
    client = chromadb.PersistentClient("./.chroma")

    doc_ids = [doc["id"] for doc in documents]
    doc_contents = [doc["content"] for doc in documents]
    doc_metadatas: list[chromadb.Metadata] = [
        {"title": doc["title"]} for doc in documents
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


def get_document_db() -> chromadb.Collection:
    client = chromadb.PersistentClient("./.chroma")

    return client.get_collection("rag")


def get_relevant_documents(
    query: str, n_results: int = 10, max_distance: float = 1.05
) -> list[Document]:
    db = get_document_db()
    result = db.query(query_texts=query, n_results=n_results)
    assert result["documents"] is not None
    assert result["distances"] is not None
    assert result["metadatas"] is not None
    documents: list[tuple[Document, float]] = [
        ({"id": id, "title": cast(str, metadata["title"]), "content": content}, dist)
        for id, metadata, content, dist in zip(
            result["ids"][0],
            result["metadatas"][0],
            result["documents"][0],
            result["distances"][0],
        )
    ]
    relevant_documents: list[Document] = [
        doc for (doc, dist) in documents if dist < max_distance
    ]
    return relevant_documents
