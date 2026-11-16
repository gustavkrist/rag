from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["user", "system"]
    content: str


class Document(TypedDict):
    id: str
    title: str
    content: str
    url: str
