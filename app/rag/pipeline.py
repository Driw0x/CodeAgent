from collections.abc import Callable

from app.llm.local_llm import LocalLLM
from app.rag.prompt import SYSTEM_PROMPT, build_prompt


class RAGPipeline:
    def __init__(
        self,
        retriever: Callable[[str, int], list[dict]],
        llm: LocalLLM,
        top_k: int = 3,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        chunks = self.retriever(question, self.top_k)

        if not chunks:
            return "Aucun contexte pertinent n'a été trouvé pour répondre à cette question."

        prompt = build_prompt(question, chunks)

        return self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )