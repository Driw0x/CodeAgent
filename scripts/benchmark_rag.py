import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.memory.vector_store import load_index
from app.rag.prompt import build_context, select_context_chunks
from app.retrieval import FaissRetriever
from app.retrieval.faiss_retriever import STOPWORDS, TYPE_BONUS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "benchmarks" / "rag" / "fixture"
DEFAULT_QUESTIONS = PROJECT_ROOT / "benchmarks" / "rag" / "rag_questions.json"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "rag" / "results"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_K_VALUES = (1, 3, 5, 10, 25)
FINAL_K_VALUES = (1, 3, 5)

RERANK_STRATEGIES = {
    "rerank_only": (0.0, 1.0),
    "rrf_equal": (1.0, 1.0),
    "rrf_rerank_2x": (1.0, 2.0),
    "rrf_hybrid_2x": (2.0, 1.0),
}


def relative_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def source_matches(chunk: dict, expected: dict) -> bool:
    if relative_path(chunk["file"]) != expected["file"]:
        return False
    if expected.get("type") is not None and chunk.get("type") != expected["type"]:
        return False
    if expected.get("name") is not None and chunk.get("name") != expected["name"]:
        return False
    return True


def matched_expected_sources(retrieved_chunks: list[dict], expected_sources: list[dict]) -> set[int]:
    return {
        expected_index
        for expected_index, expected in enumerate(expected_sources)
        if any(source_matches(chunk, expected) for chunk in retrieved_chunks)
    }


def reciprocal_rank(retrieved_chunks: list[dict], expected_sources: list[dict], k: int) -> float:
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        if any(source_matches(chunk, expected) for expected in expected_sources):
            return 1.0 / rank
    return 0.0


def calculate_metrics(
    retrieved_chunks: list[dict],
    expected_sources: list[dict],
    k_values: tuple[int, ...],
) -> dict:
    metrics = {}
    for k in k_values:
        top_k = retrieved_chunks[:k]
        matched = matched_expected_sources(top_k, expected_sources)
        relevant_chunks = sum(
            any(source_matches(chunk, expected) for expected in expected_sources)
            for chunk in top_k
        )
        metrics[f"hit@{k}"] = 1.0 if matched else 0.0
        metrics[f"recall@{k}"] = len(matched) / len(expected_sources)
        metrics[f"precision@{k}"] = relevant_chunks / k
    return metrics


def serialize_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank,
            "file": relative_path(chunk["file"]),
            "type": chunk.get("type"),
            "name": chunk.get("name"),
            "distance": chunk.get("distance"),
            "lexical_score": chunk.get("lexical_score"),
            "dense_rank": chunk.get("dense_rank"),
            "lexical_rank": chunk.get("lexical_rank"),
            "hybrid_score": chunk.get("hybrid_score"),
            "rerank_score": chunk.get("rerank_score"),
            "final_score": chunk.get("final_score"),
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def m4_tokenize(text: str) -> set[str]:
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


def m4_rerank_score(question: str, chunk: dict) -> float:
    query_tokens = m4_tokenize(question)
    if not query_tokens:
        return 0.0
    name_tokens = m4_tokenize(str(chunk.get("name", "")))
    file_tokens = m4_tokenize(str(chunk.get("file", "")))
    content_tokens = m4_tokenize(str(chunk.get("content", "")))
    all_tokens = name_tokens | file_tokens | content_tokens
    coverage = len(query_tokens & all_tokens) / len(query_tokens)
    name_overlap = len(query_tokens & name_tokens)
    file_overlap = len(query_tokens & file_tokens)
    type_bonus = TYPE_BONUS.get(str(chunk.get("type", "")), 0)
    return 10 * coverage + 2 * name_overlap + file_overlap + type_bonus


def m4_rerank(question: str, candidates: list[dict], k: int = 5) -> list[dict]:
    ranked = [dict(chunk) for chunk in candidates]
    for chunk in ranked:
        chunk["rerank_score"] = m4_rerank_score(question, chunk)
    ranked.sort(
        key=lambda chunk: (
            -chunk["rerank_score"],
            chunk.get("distance", float("inf")),
        )
    )
    return ranked[:k]


def evaluate_question(retriever: FaissRetriever, item: dict) -> dict:
    question = item["question"]
    expected_sources = item["expected_sources"]
    dense_chunks = retriever.dense_retrieve(question, k=25)
    lexical_chunks = retriever.lexical_retrieve(question, k=25)
    hybrid_chunks = retriever.hybrid_retrieve(question, k=25)
    dense_metrics = calculate_metrics(dense_chunks, expected_sources, CANDIDATE_K_VALUES)
    lexical_metrics = calculate_metrics(lexical_chunks, expected_sources, CANDIDATE_K_VALUES)
    hybrid_metrics = calculate_metrics(hybrid_chunks, expected_sources, CANDIDATE_K_VALUES)
    dense_metrics["mrr@5"] = reciprocal_rank(dense_chunks, expected_sources, 5)
    dense_metrics["mrr@25"] = reciprocal_rank(dense_chunks, expected_sources, 25)
    lexical_metrics["mrr@25"] = reciprocal_rank(lexical_chunks, expected_sources, 25)
    hybrid_metrics["mrr@25"] = reciprocal_rank(hybrid_chunks, expected_sources, 25)
    m4_chunks = m4_rerank(question, dense_chunks, k=5)
    m4_metrics = calculate_metrics(m4_chunks, expected_sources, FINAL_K_VALUES)
    m4_metrics["mrr@5"] = reciprocal_rank(m4_chunks, expected_sources, 5)
    if not dense_metrics["hit@25"]:
        m4_diagnosis = "candidate_retrieval"
    elif not m4_metrics["hit@5"]:
        m4_diagnosis = "reranking"
    else:
        m4_diagnosis = "ok"
    reranking_results = {}
    reranking_chunks = {}
    for name, (hybrid_weight, rerank_weight) in RERANK_STRATEGIES.items():
        chunks = retriever.rerank_candidates(
            question,
            hybrid_chunks,
            k=5,
            hybrid_weight=hybrid_weight,
            rerank_weight=rerank_weight,
        )
        metrics = calculate_metrics(chunks, expected_sources, FINAL_K_VALUES)
        metrics["mrr@5"] = reciprocal_rank(chunks, expected_sources, 5)
        reranking_chunks[name] = chunks
        reranking_results[name] = {
            "retrieved_sources": serialize_chunks(chunks),
            "metrics": metrics,
        }
    final_chunks = reranking_chunks["rrf_equal"]
    final_metrics = reranking_results["rrf_equal"]["metrics"]
    if not hybrid_metrics["hit@25"]:
        m5_reranking_diagnosis = "candidate_retrieval"
    elif not final_metrics["hit@5"]:
        m5_reranking_diagnosis = "reranking"
    else:
        m5_reranking_diagnosis = "ok"
    context_chunks = select_context_chunks(final_chunks)
    context_metrics = calculate_metrics(context_chunks, expected_sources, FINAL_K_VALUES)
    context_metrics["mrr@5"] = reciprocal_rank(context_chunks, expected_sources, 5)
    context_metrics["chunk_count"] = len(context_chunks)
    context_metrics["context_chars"] = len(build_context(context_chunks))
    context_metrics["dropped_chunks"] = len(final_chunks) - len(context_chunks)
    relevant_before = matched_expected_sources(final_chunks, expected_sources)
    relevant_after = matched_expected_sources(context_chunks, expected_sources)
    context_metrics["dropped_relevant"] = len(relevant_before - relevant_after)
    if not hybrid_metrics["hit@25"]:
        m5_context_diagnosis = "candidate_retrieval"
    elif not final_metrics["hit@5"]:
        m5_context_diagnosis = "reranking"
    elif not context_metrics["hit@5"]:
        m5_context_diagnosis = "context_selection"
    else:
        m5_context_diagnosis = "ok"
    return {
        "id": item["id"],
        "question": question,
        "expected_sources": expected_sources,
        "dense": {
            "retrieved_sources": serialize_chunks(dense_chunks),
            "metrics": dense_metrics,
        },
        "lexical": {
            "retrieved_sources": serialize_chunks(lexical_chunks),
            "metrics": lexical_metrics,
        },
        "hybrid": {
            "retrieved_sources": serialize_chunks(hybrid_chunks),
            "metrics": hybrid_metrics,
        },
        "m4": {
            "retrieved_sources": serialize_chunks(m4_chunks),
            "metrics": m4_metrics,
            "diagnosis": m4_diagnosis,
        },
        "m5_reranking": {
            "final": reranking_results["rrf_equal"],
            "strategies": reranking_results,
            "diagnosis": m5_reranking_diagnosis,
        },
        "m5_context": {
            "retrieved_sources": serialize_chunks(context_chunks),
            "metrics": context_metrics,
            "diagnosis": m5_context_diagnosis,
        },
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_metrics(results: list[dict], getter) -> dict:
    first_metrics = getter(results[0])
    return {
        metric: mean([getter(result)[metric] for result in results])
        for metric in first_metrics
    }


def aggregate_diagnosis(results: list[dict], getter) -> dict:
    counts = {
        "ok": 0,
        "candidate_retrieval": 0,
        "reranking": 0,
        "context_selection": 0,
    }
    for result in results:
        counts[getter(result)] += 1
    return counts


def aggregate_reranking_strategies(results: list[dict]) -> dict:
    aggregates = {}
    for strategy in RERANK_STRATEGIES:
        aggregates[strategy] = aggregate_metrics(
            results,
            lambda result, strategy=strategy:
                result["m5_reranking"]["strategies"][strategy]["metrics"],
        )
    return aggregates


def load_questions(path: Path) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("The benchmark must contain a non-empty JSON list.")
    for item in questions:
        if not item.get("id"):
            raise ValueError("Each benchmark item must have an id.")
        if not item.get("question"):
            raise ValueError(f"{item.get('id', '<unknown>')}: missing question.")
        if not item.get("expected_sources"):
            raise ValueError(f"{item['id']}: expected_sources must contain at least one source.")
    return questions


def load_retriever():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Loading frozen RAG fixture...")
    index, chunks = load_index(FIXTURE_DIR)
    print(f"Loaded {len(chunks)} frozen chunks.")
    return FaissRetriever(model=model, index=index, chunks=chunks), len(chunks)


def print_summary(
    question_count: int,
    chunk_count: int,
    m4_raw: dict,
    m4_final: dict,
    m5_final: dict,
    context: dict,
    m4_diagnosis: dict,
    m5_diagnosis: dict,
    context_diagnosis: dict,
    total_dropped_relevant: int,
) -> None:
    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"Questions : {question_count}")
    print(f"Chunks    : {chunk_count}")
    print(f"Model     : {EMBEDDING_MODEL}")
    print("Corpus    : frozen RAG fixture")
    print("\n---------- M4 BASELINE ----------")
    print(f"Dense Hit@25   : {m4_raw['hit@25']:.4f}")
    print(f"Dense Recall@25: {m4_raw['recall@25']:.4f}")
    print(f"Final Hit@1    : {m4_final['hit@1']:.4f}")
    print(f"Final Hit@3    : {m4_final['hit@3']:.4f}")
    print(f"Final Hit@5    : {m4_final['hit@5']:.4f}")
    print(f"Final Recall@5 : {m4_final['recall@5']:.4f}")
    print(f"Final MRR@5    : {m4_final['mrr@5']:.4f}")
    print(f"Diagnosis      : {m4_diagnosis}")
    print("\n---------- M5 RERANKING ----------")
    print(f"Hit@1    : {m5_final['hit@1']:.4f}")
    print(f"Hit@3    : {m5_final['hit@3']:.4f}")
    print(f"Hit@5    : {m5_final['hit@5']:.4f}")
    print(f"Recall@5 : {m5_final['recall@5']:.4f}")
    print(f"MRR@5    : {m5_final['mrr@5']:.4f}")
    print(f"Diagnosis: {m5_diagnosis}")
    print("\n---------- M5 CONTEXT ----------")
    print(f"Hit@1                  : {context['hit@1']:.4f}")
    print(f"Hit@3                  : {context['hit@3']:.4f}")
    print(f"Hit@5                  : {context['hit@5']:.4f}")
    print(f"Recall@5               : {context['recall@5']:.4f}")
    print(f"MRR@5                  : {context['mrr@5']:.4f}")
    print(f"Average chunks kept    : {context['chunk_count']:.2f}")
    print(f"Average context chars  : {context['context_chars']:.2f}")
    print(f"Average dropped chunks : {context['dropped_chunks']:.2f}")
    print(f"Total dropped relevant : {total_dropped_relevant}")
    print(f"Diagnosis               : {context_diagnosis}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CodeAgent RAG.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args()
    questions = load_questions(args.questions)
    retriever, chunk_count = load_retriever()
    print(f"\nRunning {len(questions)} benchmark questions...\n")
    results = []
    for item in questions:
        result = evaluate_question(retriever, item)
        results.append(result)
        print(
            f"{result['id']} | "
            f"M4 H@5={result['m4']['metrics']['hit@5']:.0f} | "
            f"M5 H@5={result['m5_reranking']['final']['metrics']['hit@5']:.0f} | "
            f"Context H@5={result['m5_context']['metrics']['hit@5']:.0f}"
        )
    dense_aggregate = aggregate_metrics(
        results,
        lambda result: result["dense"]["metrics"],
    )
    lexical_aggregate = aggregate_metrics(
        results,
        lambda result: result["lexical"]["metrics"],
    )
    hybrid_aggregate = aggregate_metrics(
        results,
        lambda result: result["hybrid"]["metrics"],
    )
    m4_aggregate = aggregate_metrics(results, lambda result: result["m4"]["metrics"])
    m5_final_aggregate = aggregate_metrics(
        results,
        lambda result: result["m5_reranking"]["final"]["metrics"],
    )
    context_aggregate = aggregate_metrics(
        results,
        lambda result: result["m5_context"]["metrics"],
    )
    reranking_strategies = aggregate_reranking_strategies(results)
    m4_diagnosis = aggregate_diagnosis(
        results,
        lambda result: result["m4"]["diagnosis"],
    )
    m5_diagnosis = aggregate_diagnosis(
        results,
        lambda result: result["m5_reranking"]["diagnosis"],
    )
    context_diagnosis = aggregate_diagnosis(
        results,
        lambda result: result["m5_context"]["diagnosis"],
    )
    total_dropped_relevant = sum(
        result["m5_context"]["metrics"]["dropped_relevant"]
        for result in results
    )
    print_summary(
        question_count=len(results),
        chunk_count=chunk_count,
        m4_raw=dense_aggregate,
        m4_final=m4_aggregate,
        m5_final=m5_final_aggregate,
        context=context_aggregate,
        m4_diagnosis=m4_diagnosis,
        m5_diagnosis=m5_diagnosis,
        context_diagnosis=context_diagnosis,
        total_dropped_relevant=total_dropped_relevant,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "question_count": len(results),
        "chunk_count": chunk_count,
        "fixture": relative_path(FIXTURE_DIR),
    }
    m4_payload = {
        **common,
        "pipeline": "dense_faiss + lexical_reranking",
        "aggregate": {
            "dense": dense_aggregate,
            "final": m4_aggregate,
            "diagnosis": m4_diagnosis,
        },
        "results": [
            {
                "id": result["id"],
                "question": result["question"],
                "expected_sources": result["expected_sources"],
                "dense": result["dense"],
                "final": result["m4"],
            }
            for result in results
        ],
    }
    m5_reranking_payload = {
        **common,
        "pipeline": "dense + lexical + hybrid + RRF",
        "aggregate": {
            "dense": dense_aggregate,
            "lexical": lexical_aggregate,
            "hybrid": hybrid_aggregate,
            "final": m5_final_aggregate,
            "reranking_strategies": reranking_strategies,
            "diagnosis": m5_diagnosis,
        },
        "results": [
            {
                "id": result["id"],
                "question": result["question"],
                "expected_sources": result["expected_sources"],
                "dense": result["dense"],
                "lexical": result["lexical"],
                "hybrid": result["hybrid"],
                "final": result["m5_reranking"]["final"],
                "reranking_strategies": result["m5_reranking"]["strategies"],
                "diagnosis": result["m5_reranking"]["diagnosis"],
            }
            for result in results
        ],
    }
    m5_context_payload = {
        **common,
        "pipeline": "M5 retrieval + context selection",
        "aggregate": {
            "final": m5_final_aggregate,
            "context": context_aggregate,
            "total_dropped_relevant": total_dropped_relevant,
            "diagnosis": context_diagnosis,
        },
        "results": [
            {
                "id": result["id"],
                "question": result["question"],
                "expected_sources": result["expected_sources"],
                "final": result["m5_reranking"]["final"],
                "context": result["m5_context"],
            }
            for result in results
        ],
    }
    write_json(RESULTS_DIR / "m4_baseline.json", m4_payload)
    write_json(RESULTS_DIR / "m5_reranking.json", m5_reranking_payload)
    write_json(RESULTS_DIR / "m5_context.json", m5_context_payload)


if __name__ == "__main__":
    main()
