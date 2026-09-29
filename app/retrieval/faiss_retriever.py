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

RRF_K = 60


def tokenize(text: str) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
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


def lexical_score(question: str, chunk: dict) -> float:
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
    return 10 * coverage + 2 * name_overlap + file_overlap


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
    return 10 * coverage + 2 * name_overlap + file_overlap + type_bonus


def chunk_key(chunk: dict) -> tuple:
    return (
        str(chunk.get("file", "")),
        chunk.get("type"),
        chunk.get("name"),
        chunk.get("start_line"),
        chunk.get("end_line"),
    )


class FaissRetriever:
    def __init__(self, model, index, chunks: list[dict]):
        self.model = model
        self.index = index
        self.chunks = chunks

    def _validate(self, question: str, k: int) -> None:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if k <= 0:
            raise ValueError("k must be greater than 0.")

    def dense_retrieve(self, question: str, k: int = 25) -> list[dict]:
        self._validate(question, k)

        if not self.chunks:
            return []

        query_vector = self.model.encode([question], show_progress_bar=False)
        query_vector = np.asarray(query_vector, dtype="float32")
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_vector, k)
        results = []

        for distance, index in zip(distances[0], indices[0]):
            if index < 0 or index >= len(self.chunks):
                continue
            chunk = dict(self.chunks[index])
            chunk["distance"] = float(distance)
            results.append(chunk)

        return results

    def lexical_retrieve(self, question: str, k: int = 25) -> list[dict]:
        self._validate(question, k)

        if not self.chunks:
            return []

        results = []

        for source_chunk in self.chunks:
            score = lexical_score(question, source_chunk)
            if score <= 0:
                continue
            chunk = dict(source_chunk)
            chunk["lexical_score"] = score
            results.append(chunk)

        results.sort(
            key=lambda chunk: (
                -chunk["lexical_score"],
                str(chunk.get("file", "")),
                str(chunk.get("name", "")),
            )
        )

        return results[:k]

    def hybrid_retrieve(self, question: str, k: int = 25) -> list[dict]:
        self._validate(question, k)

        if not self.chunks:
            return []

        candidate_k = min(k, len(self.chunks))
        dense = self.dense_retrieve(question, candidate_k)
        lexical = self.lexical_retrieve(question, candidate_k)
        merged = {}

        for rank, chunk in enumerate(dense, start=1):
            item = dict(chunk)
            item["dense_rank"] = rank
            item["lexical_rank"] = None
            item["lexical_score"] = lexical_score(question, item)
            item["hybrid_score"] = 1.0 / (RRF_K + rank)
            merged[chunk_key(item)] = item

        for rank, chunk in enumerate(lexical, start=1):
            key = chunk_key(chunk)
            contribution = 1.0 / (RRF_K + rank)

            if key in merged:
                merged[key]["lexical_rank"] = rank
                merged[key]["lexical_score"] = chunk["lexical_score"]
                merged[key]["hybrid_score"] += contribution
            else:
                item = dict(chunk)
                item["distance"] = None
                item["dense_rank"] = None
                item["lexical_rank"] = rank
                item["hybrid_score"] = contribution
                merged[key] = item

        candidates = list(merged.values())
        candidates.sort(
            key=lambda chunk: (
                -chunk["hybrid_score"],
                -chunk["lexical_score"],
                chunk["distance"]
                if chunk["distance"] is not None
                else float("inf"),
            )
        )

        return candidates[:k]

    def retrieve(self, question: str, k: int = 5) -> list[dict]:
        self._validate(question, k)

        if not self.chunks:
            return []

        candidate_k = min(max(k * 5, 25), len(self.chunks))
        candidates = self.hybrid_retrieve(question, candidate_k)

        for chunk in candidates:
            chunk["rerank_score"] = rerank_score(question, chunk)

        reranked = sorted(
            candidates,
            key=lambda chunk: (
                -chunk["rerank_score"],
                -chunk["hybrid_score"],
            ),
        )
        rerank_ranks = {chunk_key(chunk): rank for rank, chunk in enumerate(reranked, start=1)}

        for hybrid_rank, chunk in enumerate(candidates, start=1):
            rerank_rank = rerank_ranks[chunk_key(chunk)]
            chunk["final_score"] = 1.0 / (RRF_K + hybrid_rank) + 1.0 / (RRF_K + rerank_rank)

        candidates.sort(
            key=lambda chunk: (
                -chunk["final_score"],
                -chunk["rerank_score"],
                -chunk["hybrid_score"],
            )
        )

        return candidates[:k]

    def rerank_candidates(
        self,
        question: str,
        candidates: list[dict],
        k: int = 5,
        hybrid_weight: float = 1.0,
        rerank_weight: float = 1.0,
    ) -> list[dict]:
        ranked = [dict(chunk) for chunk in candidates]

        for chunk in ranked:
            chunk["rerank_score"] = rerank_score(question, chunk)

        reranked = sorted(
            ranked,
            key=lambda chunk: (
                -chunk["rerank_score"],
                -chunk["hybrid_score"],
            ),
        )
        hybrid_ranks = {chunk_key(chunk): rank for rank, chunk in enumerate(ranked, start=1)}
        rerank_ranks = {chunk_key(chunk): rank for rank, chunk in enumerate(reranked, start=1)}

        for chunk in ranked:
            key = chunk_key(chunk)
            chunk["final_score"] = (
                hybrid_weight / (RRF_K + hybrid_ranks[key])
                + rerank_weight / (RRF_K + rerank_ranks[key])
            )

        ranked.sort(
            key=lambda chunk: (
                -chunk["final_score"],
                -chunk["rerank_score"],
                -chunk["hybrid_score"],
            )
        )

        return ranked[:k]
