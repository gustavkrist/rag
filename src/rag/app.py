import logging
import textwrap

import gradio as gr

from rag.documents import DocumentStore
from rag.llm import RagLLM
from rag.types import Message

logger = logging.getLogger(__name__)


class App:
    def __init__(self, document_path: str = "data/articles.json") -> None:
        document_db = DocumentStore(document_path)
        self.llm = RagLLM(document_db)

    def _response(self, message: str, _history: list[Message]) -> str:
        reply, used_documents = self.llm.generate_reply(message)
        doc_entries = [f"[{d['title']}]({d['url']})" for d in used_documents]
        articles_text = textwrap.dedent(f"""\


        ---
        Used articles:
        - {"\n        - ".join(doc_entries)}""")
        return reply + (articles_text if used_documents else "")

    def run(self) -> None:
        app = gr.ChatInterface(
            fn=self._response,
            title="Terraria Wiki Helper",
            description="Ask questions about progression, bosses, strageties, and more",
            examples=[],
        )
        app.launch()


def run_app() -> None:
    app = App()
    app.run()
