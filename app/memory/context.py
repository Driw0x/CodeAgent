"""Conservative context accounting, shared by RAG, tools and Ollama."""

import json
import os


DEFAULT_CONTEXT_TOKENS = 16_384


def estimate_tokens(value) -> int:
    """One token per UTF-8 byte, plus framing: an intentionally high estimate.

    No tokenizer/model download. This is not Ollama's actual token count.
    JSON serialization also accounts for roles, tool calls and schema fields.
    """
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return len(text.encode("utf-8"))


class ContextBudget:
    def __init__(self, context_tokens: int = DEFAULT_CONTEXT_TOKENS, response_tokens: int = 512):
        if context_tokens <= 0 or response_tokens <= 0 or context_tokens <= response_tokens + 256:
            raise ValueError("Context must exceed response reserve plus 256 framing tokens.")
        self.context_tokens = context_tokens
        self.response_tokens = response_tokens
        self.input_limit = context_tokens - response_tokens - 256

    @classmethod
    def for_llm(cls, llm):
        context = getattr(llm, "context_tokens", None)
        response = getattr(llm, "num_predict", None)
        if not isinstance(context, int):
            context = int(os.getenv("CODEAGENT_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS))
        if not isinstance(response, int) or response <= 0:
            response = 512
        return cls(context, response)

    def measure(self, messages: list[dict], tools=None) -> int:
        return estimate_tokens(messages) + 32 * len(messages) + (estimate_tokens(tools) if tools else 0)

    def check(self, messages: list[dict], tools=None) -> int:
        size = self.measure(messages, tools)
        if size > self.input_limit:
            raise RuntimeError(
                f"Context budget exceeded: {size} estimated tokens > {self.input_limit}; "
                "shorten the question or request a smaller source window."
            )
        return size


def reduce_observation(content: str, question: str, max_bytes: int = 2500) -> str:
    """Select whole JSON items/lines; never truncate JSON mid-field.

    Paths, line numbers and scalar identifiers stay attached to selected items.
    The original remains in the source/JSONL archive, outside the LLM context.
    """
    if estimate_tokens(content) <= max_bytes:
        return content
    from app.retrieval.faiss_retriever import tokenize

    query = tokenize(question)

    def score(value):
        return len(query & tokenize(str(value)))

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"content": content}
    if not isinstance(data, dict):
        data = {"result": data}
    text_fields = {"content", "diff", "stdout", "stderr", "result"}
    result = {key: value for key, value in data.items() if not isinstance(value, (list, dict)) and key not in text_fields}
    result["truncated"] = True
    result["original_bytes"] = estimate_tokens(content)

    for key, value in data.items():
        if isinstance(value, list):
            selected = []
            for _, item in sorted(enumerate(value), key=lambda pair: (-score(pair[1]), pair[0])):
                candidate = {**result, key: [*selected, item]}
                if estimate_tokens(candidate) <= max_bytes:
                    selected.append(item)
            result[key] = selected
        elif isinstance(value, dict):
            if estimate_tokens({**result, key: value}) <= max_bytes:
                result[key] = value
        elif key == "result" and isinstance(value, str):
            allowance = max_bytes - estimate_tokens(result) - 64
            if allowance > 128:
                result[key] = json.loads(reduce_observation(value, question, allowance)) if estimate_tokens(value) > allowance else value
        elif key in text_fields and isinstance(value, str):
            lines = value.splitlines()
            base = data.get("start_line", 1)
            if not isinstance(base, int):
                base = 1
            selected = []
            for index in sorted(range(len(lines)), key=lambda index: (-score(lines[index]), index)):
                item = {"line": base + index, "text": lines[index]}
                candidate = {**result, key + "_excerpts": [*selected, item]}
                if estimate_tokens(candidate) <= max_bytes:
                    selected.append(item)
            result[key + "_excerpts"] = sorted(selected, key=lambda item: item["line"])
    reduced = json.dumps(result, ensure_ascii=False)
    if estimate_tokens(reduced) > max_bytes:
        raise RuntimeError("Observation metadata exceeds context budget; request a smaller result.")
    return reduced


def fit_sources(question, sources, builder, system, budget, history="", reduce_tools=False):
    """Fit complete RAG chunks or selected tool observations with contiguous IDs."""
    selected = []
    # Current evidence takes priority over optional conversational history in a sourced final.
    fixed = budget.measure([{"role": "system", "content": system}, {"role": "user", "content": builder(question, [])}])
    available = budget.input_limit - fixed
    if available < 0:
        budget.check([{"role": "system", "content": system}, {"role": "user", "content": builder(question, [])}])
    # Short diagnostics need only their actual size, not a full share of a long excerpt's budget.
    small = [source for source in sources if estimate_tokens(source["content"]) <= max(256, available // max(1, len(sources)))]
    large_available = available - sum(estimate_tokens(source["content"]) + 256 for source in small)
    large_count = max(1, len(sources) - len(small))
    for source in sources:
        source = dict(source)
        if reduce_tools and estimate_tokens(source["content"]) > max(256, available // max(1, len(sources))):
            header, separator, raw = source["content"].partition("Result:\n")
            if separator:
                # JSON embedded in a prompt is escaped again in the Ollama payload.
                limit = max(128, large_available // (2 * large_count) - estimate_tokens(header) - 256)
                source["content"] = header + separator + reduce_observation(raw, question, limit)
        candidate = [*selected, source]
        prompt = builder(question, candidate)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        if budget.measure(messages) <= budget.input_limit:
            selected = candidate
    prompt = builder(question, selected)
    if history:
        with_history = f"{history}\n\n{prompt}"
        if budget.measure([{"role": "system", "content": system}, {"role": "user", "content": with_history}]) <= budget.input_limit:
            prompt = with_history
    budget.check([{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    return selected, prompt
