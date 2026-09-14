import re
import unicodedata

import numpy as np


STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "comment", "dans", "de", "des",
    "du", "elle", "en", "est", "et", "ils", "la", "le", "les", "par",
    "pour", "que", "qui", "sont", "sur", "un", "une",
    "are", "how", "in", "is", "of", "the", "to",
}


TYPE_BONUS = {
    "function": 2,
    "class": 2,
    "method": 2,
    "variable": -1,
    "import": -3,
    "import_from": -3,
}


def tokenize(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    )

    normalized = normalized.replace("_", " ")

    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    tokens = set()

    for token in raw_tokens:
        if token in STOPWORDS:
            continue

        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]

        tokens.add(token)

    return tokens


def rerank_score(question: str, chunk: dict) -> float:
    query_tokens = tokenize(question)

    if not query_tokens:
        return 0.0

    name_tokens = tokenize(str(chunk.get("name", "")))
    file_tokens = tokenize(str(chunk.get("file", "")))
    content_tokens = tokenize(str(chunk.get("content", "")))

    all_tokens = name_tokens | file_tokens | content_tokens

    coverage = len(query_tokens & all_tokens) / len(query_tokens)
    name_overlap = len(query_tokens & name_tokens)
    file_overlap = len(query_tokens & file_tokens)
    type_bonus = TYPE_BONUS.get(str(chunk.get("type", "")), 0)

    return (
        10 * coverage
        + 2 * name_overlap
        + file_overlap
        + type_bonus
    )


class FaissRetriever:
    def __init__(self, model, index, chunks: list[dict]):
        self.model = model
        self.index = index
        self.chunks = chunks

    def retrieve(self, question: str, k: int = 5) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if not self.chunks:
            return []

        query_vector = self.model.encode(
            [question],
            show_progress_bar=False,
        )

        query_vector = np.asarray(
            query_vector,
            dtype="float32",
        )

        candidate_k = min(
            max(k * 5, 25),
            len(self.chunks),
        )

        distances, indices = self.index.search(
            query_vector,
            candidate_k,
        )

        candidates = []

        for distance, index in zip(
            distances[0],
            indices[0],
        ):
            if index < 0 or index >= len(self.chunks):
                continue

            chunk = dict(self.chunks[index])

            chunk["distance"] = float(distance)
            chunk["rerank_score"] = rerank_score(
                question,
                chunk,
            )

            candidates.append(chunk)

        candidates.sort(
            key=lambda chunk: (
                -chunk["rerank_score"],
                chunk["distance"],
            )
        )

        return candidates[:k]