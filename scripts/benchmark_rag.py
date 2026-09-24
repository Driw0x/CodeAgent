import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.memory import build_index
from app.parser import chunking, read_dir
from app.retrieval import FaissRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "benchmarks" / "rag" / "rag_questions.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "rag" / "results"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

RAW_K_VALUES = (1, 3, 5, 10, 25)
RERANKED_K_VALUES = (1, 3, 5)


def relative_path(path: str | Path) -> str:
    path = Path(path)

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def source_matches(chunk: dict, expected: dict) -> bool:
    if relative_path(chunk["file"]) != expected["file"]:
        return False

    expected_type = expected.get("type")
    if expected_type is not None and chunk.get("type") != expected_type:
        return False

    expected_name = expected.get("name")
    if expected_name is not None and chunk.get("name") != expected_name:
        return False

    return True


def matched_expected_sources(
    retrieved_chunks: list[dict],
    expected_sources: list[dict],
) -> set[int]:
    matched = set()

    for expected_index, expected in enumerate(expected_sources):
        if any(source_matches(chunk, expected) for chunk in retrieved_chunks):
            matched.add(expected_index)

    return matched


def reciprocal_rank(
    retrieved_chunks: list[dict],
    expected_sources: list[dict],
    k: int,
) -> float:
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


def raw_faiss_retrieve(
    retriever: FaissRetriever,
    question: str,
    k: int,
) -> list[dict]:
    query_vector = retriever.model.encode(
        [question],
        show_progress_bar=False,
    )

    query_vector = np.asarray(query_vector, dtype="float32")
    k = min(k, len(retriever.chunks))

    distances, indices = retriever.index.search(query_vector, k)
    results = []

    for distance, index in zip(distances[0], indices[0]):
        if index < 0 or index >= len(retriever.chunks):
            continue

        chunk = dict(retriever.chunks[index])
        chunk["distance"] = float(distance)
        results.append(chunk)

    return results


def serialize_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank,
            "file": relative_path(chunk["file"]),
            "type": chunk.get("type"),
            "name": chunk.get("name"),
            "distance": chunk.get("distance"),
            "rerank_score": chunk.get("rerank_score"),
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def evaluate_question(
    retriever: FaissRetriever,
    item: dict,
) -> dict:
    question = item["question"]
    expected_sources = item["expected_sources"]

    raw_chunks = raw_faiss_retrieve(
        retriever,
        question,
        max(RAW_K_VALUES),
    )

    reranked_chunks = retriever.retrieve(
        question,
        k=max(RERANKED_K_VALUES),
    )

    raw_metrics = calculate_metrics(
        raw_chunks,
        expected_sources,
        RAW_K_VALUES,
    )

    reranked_metrics = calculate_metrics(
        reranked_chunks,
        expected_sources,
        RERANKED_K_VALUES,
    )

    raw_metrics["mrr@5"] = reciprocal_rank(
        raw_chunks,
        expected_sources,
        5,
    )

    raw_metrics["mrr@25"] = reciprocal_rank(
        raw_chunks,
        expected_sources,
        25,
    )

    reranked_metrics["mrr@5"] = reciprocal_rank(
        reranked_chunks,
        expected_sources,
        5,
    )

    if not raw_metrics["hit@25"]:
        diagnosis = "candidate_retrieval"
    elif not reranked_metrics["hit@5"]:
        diagnosis = "reranking"
    else:
        diagnosis = "ok"

    return {
        "id": item["id"],
        "question": question,
        "expected_sources": expected_sources,
        "raw_faiss": {
            "retrieved_sources": serialize_chunks(raw_chunks),
            "metrics": raw_metrics,
        },
        "reranked": {
            "retrieved_sources": serialize_chunks(reranked_chunks),
            "metrics": reranked_metrics,
        },
        "diagnosis": diagnosis,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_metrics(results: list[dict], mode: str) -> dict:
    metric_names = results[0][mode]["metrics"].keys()

    return {
        metric: mean(
            [result[mode]["metrics"][metric] for result in results]
        )
        for metric in metric_names
    }


def aggregate_diagnosis(results: list[dict]) -> dict:
    counts = {
        "ok": 0,
        "candidate_retrieval": 0,
        "reranking": 0,
    }

    for result in results:
        counts[result["diagnosis"]] += 1

    return counts


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
            raise ValueError(
                f"{item['id']}: expected_sources must contain at least one source."
            )

    return questions


def build_retriever():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Reading project files...")
    files = read_dir(PROJECT_ROOT)

    chunks = []
    for file in files:
        chunks.extend(chunking(file))

    print(f"Building FAISS index from {len(chunks)} chunks...")

    index = build_index(
        model=model,
        chunks=chunks,
        dimension=EMBEDDING_DIMENSION,
    )

    retriever = FaissRetriever(
        model=model,
        index=index,
        chunks=chunks,
    )

    return retriever, len(chunks)


def print_question_result(result: dict) -> None:
    raw = result["raw_faiss"]["metrics"]
    reranked = result["reranked"]["metrics"]

    print(
        f"{result['id']} | "
        f"Raw H@5={raw['hit@5']:.0f} "
        f"H@25={raw['hit@25']:.0f} | "
        f"Rerank H@1={reranked['hit@1']:.0f} "
        f"H@5={reranked['hit@5']:.0f} | "
        f"{result['diagnosis']}"
    )


def print_summary(
    question_count: int,
    chunk_count: int,
    raw: dict,
    reranked: dict,
    diagnosis: dict,
) -> None:
    print("\n========== RAG RETRIEVAL BENCHMARK ==========")
    print(f"Questions : {question_count}")
    print(f"Chunks    : {chunk_count}")
    print(f"Model     : {EMBEDDING_MODEL}")

    print("\n---------- RAW FAISS ----------")

    for k in RAW_K_VALUES:
        print(f"Hit@{k:<2}       : {raw[f'hit@{k}']:.4f}")
        print(f"Recall@{k:<2}    : {raw[f'recall@{k}']:.4f}")

    print(f"MRR@5       : {raw['mrr@5']:.4f}")
    print(f"MRR@25      : {raw['mrr@25']:.4f}")

    print("\n------ FAISS + RERANKING ------")

    for k in RERANKED_K_VALUES:
        print(f"Hit@{k:<2}       : {reranked[f'hit@{k}']:.4f}")
        print(f"Recall@{k:<2}    : {reranked[f'recall@{k}']:.4f}")

    print(f"MRR@5       : {reranked['mrr@5']:.4f}")

    print("\n----------- DELTA ------------")

    for k in (1, 3, 5):
        delta = reranked[f"hit@{k}"] - raw[f"hit@{k}"]
        print(f"Hit@{k} delta : {delta:+.4f}")

    mrr_delta = reranked["mrr@5"] - raw["mrr@5"]
    print(f"MRR@5 delta : {mrr_delta:+.4f}")

    print("\n--------- DIAGNOSIS ----------")
    print(f"OK                  : {diagnosis['ok']}")
    print(f"Candidate retrieval : {diagnosis['candidate_retrieval']}")
    print(f"Reranking           : {diagnosis['reranking']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CodeAgent RAG retrieval."
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()
    questions = load_questions(args.questions)
    retriever, chunk_count = build_retriever()

    print(f"\nRunning {len(questions)} benchmark questions...\n")

    results = []

    for item in questions:
        result = evaluate_question(retriever, item)
        results.append(result)
        print_question_result(result)

    raw_aggregate = aggregate_metrics(results, "raw_faiss")
    reranked_aggregate = aggregate_metrics(results, "reranked")
    diagnosis = aggregate_diagnosis(results)

    print_summary(
        question_count=len(results),
        chunk_count=chunk_count,
        raw=raw_aggregate,
        reranked=reranked_aggregate,
        diagnosis=diagnosis,
    )

    output = args.output

    if output is None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = DEFAULT_RESULTS_DIR / "baseline_m4.json"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "question_count": len(results),
        "chunk_count": chunk_count,
        "raw_k_values": list(RAW_K_VALUES),
        "reranked_k_values": list(RERANKED_K_VALUES),
        "aggregate": {
            "raw_faiss": raw_aggregate,
            "reranked": reranked_aggregate,
            "diagnosis": diagnosis,
        },
        "results": results,
    }

    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()