from collections.abc import Callable

from app.llm.local_llm import LocalLLM
from app.rag.citations import extract_citations, validate_citations
from app.rag.grounding import verify_grounding
from app.rag.prompt import SYSTEM_PROMPT, build_prompt, format_sources, select_context_chunks


class RAGPipeline:
    def __init__(self, retriever: Callable[[str, int], list[dict]], llm: LocalLLM, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        chunks = self.retriever(question, self.top_k)

        if not chunks:
            return "Aucun contexte pertinent n'a été trouvé pour répondre à cette question."

        context_chunks = select_context_chunks(chunks)

        if not context_chunks:
            return "Aucun contexte pertinent n'a été trouvé pour répondre à cette question."

        prompt = build_prompt(question, context_chunks)
        answer = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        if not validate_citations(answer, len(context_chunks)):
            return "La réponse générée contient des citations invalides ou manquantes."

        if extract_citations(answer) and not verify_grounding(answer, context_chunks, self.llm):
            return "La réponse générée contient des affirmations non supportées par les sources."

        sources = format_sources(context_chunks)
        return f"{answer}\n\nSources:\n{sources}"
