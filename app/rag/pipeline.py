from collections.abc import Callable

from app.llm.local_llm import LocalLLM
from app.memory.context import ContextBudget, fit_sources
from app.memory.conversation import ConversationMemory
from app.rag.citations import extract_citations, validate_citations
from app.rag.grounding import claims_have_citations, verify_grounding
from app.rag.prompt import SYSTEM_PROMPT, build_prompt, format_sources, select_context_chunks


class RAGPipeline:
    def __init__(self, retriever: Callable[[str, int], list[dict]], llm: LocalLLM, top_k: int = 5, memory=None):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.budget = ContextBudget.for_llm(llm)
        self.memory = memory if memory is not None else ConversationMemory(max_tokens=max(512, self.budget.input_limit // 3))

    def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            answer = self._answer(question)
        except RuntimeError as error:
            self.memory.record(question, f"Erreur non résolue : {error}")
            raise
        self.memory.record(question, answer)
        return answer

    def _answer(self, question: str) -> str:
        self.last_context_stats = {}
        self.budget.check([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}])

        chunks = self.retriever(self.memory.retrieval_question(question), self.top_k)

        if not chunks:
            return "Aucun contexte pertinent n'a été trouvé pour répondre à cette question."

        context_chunks = select_context_chunks(chunks)

        if not context_chunks:
            return "Aucun contexte pertinent n'a été trouvé pour répondre à cette question."

        history = self.memory.recall(question, self.budget.input_limit // 4)
        context_chunks, prompt = fit_sources(question, context_chunks, build_prompt, SYSTEM_PROMPT, self.budget, history)
        if not context_chunks:
            return "Aucun contexte pertinent n'a été trouvé pour répondre à cette question."
        self.last_context_stats = {"estimated_tokens": self.budget.measure([
            {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}
        ]), "input_limit": self.budget.input_limit, "compressions": self.memory.compressions}
        answer = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        if not validate_citations(answer, len(context_chunks)) or not claims_have_citations(answer):
            return "La réponse générée contient des citations invalides ou manquantes."

        if extract_citations(answer) and not verify_grounding(answer, context_chunks, self.llm):
            return "La réponse générée contient des affirmations non supportées par les sources."

        sources = format_sources(context_chunks)
        return f"{answer}\n\nSources:\n{sources}"
