import argparse
import json
import math
import re
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from sentence_transformers import SentenceTransformer

from app.memory.vector_store import load_index
from app.rag.prompt import build_context, select_context_chunks
from app.retrieval import FaissRetriever
from app.retrieval.faiss_retriever import STOPWORDS, TYPE_BONUS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "benchmarks" / "rag" / "fixture"
DEFAULT_QUESTIONS = PROJECT_ROOT / "benchmarks" / "rag" / "rag_questions.json"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "rag" / "results"
M5_BASELINE = RESULTS_DIR / "m5_reranking.json"

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
        normalized = str(path).replace("\\", "/")
        marker = f"/{PROJECT_ROOT.name}/"
        position = normalized.casefold().rfind(marker.casefold())
        return normalized[position + len(marker):] if position >= 0 else normalized


def source_matches(chunk: dict, expected: dict) -> bool:
    actual_file = relative_path(chunk["file"]).replace("\\", "/")
    expected_file = expected["file"].replace("\\", "/")
    if actual_file != expected_file and not actual_file.endswith("/" + expected_file):
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


def ndcg(retrieved_chunks: list[dict], expected_sources: list[dict], k: int) -> float:
    matched = set()
    score = 0.0
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        found = next((index for index, expected in enumerate(expected_sources)
                      if index not in matched and source_matches(chunk, expected)), None)
        if found is not None:
            matched.add(found)
            score += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(expected_sources)) + 1))
    return score / ideal if ideal else 0.0


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
        metrics[f"ndcg@{k}"] = ndcg(retrieved_chunks, expected_sources, k)
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
            "bm25_score": chunk.get("bm25_score"),
            "specialized_score": chunk.get("specialized_score"),
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def m4_tokenize(text: str) -> set[str]:
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
    ranked.sort(key=lambda chunk: (-chunk["rerank_score"], chunk.get("distance", float("inf"))))
    return ranked[:k]


def specialized_rerank(model, question: str, candidates: list[dict], k: int = 5) -> list[dict]:
    ranked = [dict(chunk) for chunk in candidates]
    pairs = [
        (question, f"{chunk.get('file', '')}\n{chunk.get('name', '')}\n{chunk.get('content', '')}")
        for chunk in ranked
    ]
    for chunk, score in zip(ranked, model.predict(pairs, show_progress_bar=False)):
        chunk["specialized_score"] = float(score)
    ranked.sort(key=lambda chunk: -chunk["specialized_score"])
    return ranked[:k]


def local_reranker_features(question: str, chunk: dict) -> list[float]:
    query = m4_tokenize(question)
    name = m4_tokenize(str(chunk.get("name", "")))
    file = m4_tokenize(str(chunk.get("file", "")))
    content = m4_tokenize(str(chunk.get("content", "")))
    all_tokens = name | file | content
    return [
        float(chunk.get("bm25_score", 0.0)),
        len(query & all_tokens) / len(query) if query else 0.0,
        float(len(query & name)),
        float(len(query & file)),
        float(TYPE_BONUS.get(str(chunk.get("type", "")), 0)),
        float(len(content)),
    ]


def train_local_reranker(retriever: FaissRetriever, questions: list[dict]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = []
    labels = []
    for item in questions:
        for chunk in retriever.bm25_retrieve(item["question"], k=25):
            features.append(local_reranker_features(item["question"], chunk))
            labels.append(any(source_matches(chunk, expected) for expected in item["expected_sources"]))
    if not any(labels) or all(labels):
        raise ValueError("Local reranker training requires positive and negative examples.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0),
    )
    model.fit(features, labels)
    return model


def local_rerank(model, question: str, candidates: list[dict], k: int = 5) -> list[dict]:
    ranked = [dict(chunk) for chunk in candidates]
    scores = model.predict_proba([local_reranker_features(question, chunk) for chunk in ranked])[:, 1]
    for chunk, score in zip(ranked, scores):
        chunk["specialized_score"] = float(score)
    ranked.sort(key=lambda chunk: -chunk["specialized_score"])
    return ranked[:k]


def evaluate_question(
    retriever: FaissRetriever,
    item: dict,
    cross_encoder=None,
    reformulated_question: str | None = None,
    retry_threshold: float | None = None,
) -> dict:
    question = item["question"]
    expected_sources = item["expected_sources"]

    timings = {}

    def measured(name, call):
        start = perf_counter()
        result = call()
        timings[name] = perf_counter() - start
        return result

    dense_chunks = measured("dense", lambda: retriever.dense_retrieve(question, k=25))
    lexical_chunks = measured("lexical", lambda: retriever.lexical_retrieve(question, k=25))
    bm25_chunks = measured("bm25", lambda: retriever.bm25_retrieve(question, k=25))
    hybrid_chunks = measured("hybrid", lambda: retriever.hybrid_retrieve(question, k=25))

    dense_metrics = calculate_metrics(dense_chunks, expected_sources, CANDIDATE_K_VALUES)
    lexical_metrics = calculate_metrics(lexical_chunks, expected_sources, CANDIDATE_K_VALUES)
    bm25_metrics = calculate_metrics(bm25_chunks, expected_sources, CANDIDATE_K_VALUES)
    hybrid_metrics = calculate_metrics(hybrid_chunks, expected_sources, CANDIDATE_K_VALUES)
    dense_metrics["mrr@5"] = reciprocal_rank(dense_chunks, expected_sources, 5)
    dense_metrics["mrr@25"] = reciprocal_rank(dense_chunks, expected_sources, 25)
    lexical_metrics["mrr@25"] = reciprocal_rank(lexical_chunks, expected_sources, 25)
    bm25_metrics["mrr@25"] = reciprocal_rank(bm25_chunks, expected_sources, 25)
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

    specialized = None
    if cross_encoder is not None:
        specialized_chunks = measured(
            "specialized", lambda: specialized_rerank(cross_encoder, question, hybrid_chunks)
        )
        specialized = {
            "retrieved_sources": serialize_chunks(specialized_chunks),
            "metrics": {
                **calculate_metrics(specialized_chunks, expected_sources, FINAL_K_VALUES),
                "mrr@5": reciprocal_rank(specialized_chunks, expected_sources, 5),
            },
            "latency_seconds": timings["specialized"],
        }

    fallback = None
    if reformulated_question is not None and retry_threshold is not None:
        fallback_chunks = measured(
            "fallback",
            lambda: retriever.retrieve_with_fallback(
                question,
                5,
                lambda _: reformulated_question,
                lambda chunks: not chunks or chunks[0].get("rerank_score", 0.0) <= retry_threshold,
            ),
        )
        fallback = {
            "triggered": retriever.last_fallback_used,
            "reformulated_question": reformulated_question,
            "retrieved_sources": serialize_chunks(fallback_chunks),
            "metrics": {
                **calculate_metrics(fallback_chunks, expected_sources, FINAL_K_VALUES),
                "mrr@5": reciprocal_rank(fallback_chunks, expected_sources, 5),
            },
            "latency_seconds": timings["fallback"],
        }

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
            "latency_seconds": timings["dense"],
        },
        "lexical": {
            "retrieved_sources": serialize_chunks(lexical_chunks),
            "metrics": lexical_metrics,
            "latency_seconds": timings["lexical"],
        },
        "bm25": {
            "retrieved_sources": serialize_chunks(bm25_chunks),
            "metrics": bm25_metrics,
            "latency_seconds": timings["bm25"],
        },
        "hybrid": {
            "retrieved_sources": serialize_chunks(hybrid_chunks),
            "metrics": hybrid_metrics,
            "latency_seconds": timings["hybrid"],
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
        "specialized": specialized,
        "fallback": fallback,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_latency(results: list[dict], key: str) -> dict:
    values = [result[key]["latency_seconds"] for result in results if result.get(key)]
    return {
        "mean_seconds": statistics.mean(values) if values else 0.0,
        "p50_seconds": percentile(values, 0.50),
        "p95_seconds": percentile(values, 0.95),
    }


def aggregate_metrics(results: list[dict], getter) -> dict:
    first_metrics = getter(results[0])
    return {
        metric: mean([getter(result)[metric] for result in results])
        for metric in first_metrics
    }


def aggregate_diagnosis(results: list[dict], getter) -> dict:
    counts = {"ok": 0, "candidate_retrieval": 0, "reranking": 0, "context_selection": 0}
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


def load_questions(path: Path, require_sources: bool = True) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("The benchmark must contain a non-empty JSON list.")
    for item in questions:
        if not item.get("id"):
            raise ValueError("Each benchmark item must have an id.")
        if not item.get("question"):
            raise ValueError(f"{item.get('id', '<unknown>')}: missing question.")
        if require_sources and not item.get("expected_sources"):
            raise ValueError(f"{item['id']}: expected_sources must contain at least one source.")
    return questions


def load_retriever():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
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


def run_bm25_only(questions: list[dict], output_dir: Path) -> None:
    questions = [item for item in questions if item.get("expected_sources")]
    chunks = json.loads((FIXTURE_DIR / "chunks.json").read_text(encoding="utf-8"))
    retriever = FaissRetriever(model=None, index=None, chunks=chunks)
    results = []
    for item in questions:
        start = perf_counter()
        retrieved = retriever.bm25_retrieve(item["question"], k=25)
        latency = perf_counter() - start
        metrics = calculate_metrics(retrieved, item["expected_sources"], CANDIDATE_K_VALUES)
        metrics["mrr@25"] = reciprocal_rank(retrieved, item["expected_sources"], 25)
        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_sources": item["expected_sources"],
            "retrieved_sources": serialize_chunks(retrieved),
            "metrics": metrics,
            "latency_seconds": latency,
        })
    baseline = None
    baseline_pipeline = None
    if M5_BASELINE.exists():
        previous = json.loads(M5_BASELINE.read_text(encoding="utf-8"))
        if previous.get("question_count") == len(questions):
            baseline = previous.get("aggregate", {}).get("lexical")
            baseline_pipeline = "m5_lexical"
    final_baseline = RESULTS_DIR / "m5_final.json"
    if baseline is None and final_baseline.exists():
        previous = json.loads(final_baseline.read_text(encoding="utf-8"))
        if previous.get("summary", {}).get("retrieval_questions") == len(questions):
            baseline = previous["summary"].get("retrieval")
            baseline_pipeline = "m5_final"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "bm25_stdlib",
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "fixture": relative_path(FIXTURE_DIR),
        "aggregate": {
            "bm25": aggregate_metrics(results, lambda result: result["metrics"]),
            "latency": aggregate_latency([{"bm25": result} for result in results], "bm25"),
            "existing_baseline": baseline,
            "existing_baseline_pipeline": baseline_pipeline,
        },
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "m9_bm25.json", payload)


def existing_holdout_baseline(question_count: int) -> dict | None:
    path = RESULTS_DIR / "m5_final.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("summary", {}).get("retrieval_questions") != question_count:
        return None
    return result["summary"].get("retrieval")


def run_local_reranker(questions: list[dict], output_dir: Path) -> None:
    training = load_questions(DEFAULT_QUESTIONS)
    evaluation = [item for item in questions if item.get("expected_sources")]
    chunks = json.loads((FIXTURE_DIR / "chunks.json").read_text(encoding="utf-8"))
    retriever = FaissRetriever(model=None, index=None, chunks=chunks)
    train_start = perf_counter()
    model = train_local_reranker(retriever, training)
    training_seconds = perf_counter() - train_start
    rows = []
    for item in evaluation:
        candidates = retriever.bm25_retrieve(item["question"], k=25)
        start = perf_counter()
        ranked = local_rerank(model, item["question"], candidates)
        latency = perf_counter() - start
        metrics = calculate_metrics(ranked, item["expected_sources"], FINAL_K_VALUES)
        metrics["mrr@5"] = reciprocal_rank(ranked, item["expected_sources"], 5)
        rows.append({
            "id": item["id"],
            "retrieved_sources": serialize_chunks(ranked),
            "metrics": metrics,
            "latency_seconds": latency,
        })
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "bm25_candidates + local_logistic_reranker",
        "training_questions": len(training),
        "holdout_questions": len(evaluation),
        "training_seconds": training_seconds,
        "aggregate": {
            "specialized": aggregate_metrics(rows, lambda row: row["metrics"]),
            "latency": aggregate_latency([{"specialized": row} for row in rows], "specialized"),
            "existing_m5_baseline": existing_holdout_baseline(len(evaluation)),
        },
        "results": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "m9_local_reranker.json", payload)


def expanded_query(question: str, candidates: list[dict]) -> str:
    names = " ".join(str(chunk.get("name", "")) for chunk in candidates[:3])
    return f"{question} {names}".strip()


def iterative_bm25_row(retriever: FaissRetriever, item: dict, threshold: float) -> dict:
    start = perf_counter()
    initial = retriever.bm25_retrieve(item["question"], k=25)
    triggered = not initial or initial[0]["bm25_score"] <= threshold
    final = retriever.bm25_retrieve(expanded_query(item["question"], initial), k=25) if triggered else initial
    latency = perf_counter() - start
    top = final[:5]
    metrics = calculate_metrics(top, item["expected_sources"], FINAL_K_VALUES)
    metrics["mrr@5"] = reciprocal_rank(top, item["expected_sources"], 5)
    return {
        "id": item["id"],
        "triggered": triggered,
        "metrics": metrics,
        "latency_seconds": latency,
        "retrieved_sources": serialize_chunks(top),
    }


def run_iterative_bm25(questions: list[dict], output_dir: Path) -> None:
    development = load_questions(DEFAULT_QUESTIONS)
    evaluation = [item for item in questions if item.get("expected_sources")]
    chunks = json.loads((FIXTURE_DIR / "chunks.json").read_text(encoding="utf-8"))
    retriever = FaissRetriever(model=None, index=None, chunks=chunks)
    top_scores = [retriever.bm25_retrieve(item["question"], k=1)[0]["bm25_score"] for item in development]
    candidates = [float("-inf"), *sorted(set(top_scores))]
    trials = []
    for threshold in candidates:
        rows = [iterative_bm25_row(retriever, item, threshold) for item in development]
        metrics = aggregate_metrics(rows, lambda row: row["metrics"])
        trials.append((metrics["recall@5"], metrics["mrr@5"], -sum(row["triggered"] for row in rows), threshold))
    _, _, _, threshold = max(trials)
    development_rows = [iterative_bm25_row(retriever, item, threshold) for item in development]
    holdout_rows = [iterative_bm25_row(retriever, item, threshold) for item in evaluation]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "bm25 + one pseudo-relevance reformulation",
        "calibration": {
            "dataset": relative_path(DEFAULT_QUESTIONS),
            "threshold": threshold if math.isfinite(threshold) else None,
            "decision": "enabled" if math.isfinite(threshold) else "disabled_no_development_gain",
            "metrics": aggregate_metrics(development_rows, lambda row: row["metrics"]),
            "triggered": sum(row["triggered"] for row in development_rows),
        },
        "holdout": {
            "questions": len(evaluation),
            "metrics": aggregate_metrics(holdout_rows, lambda row: row["metrics"]),
            "triggered": sum(row["triggered"] for row in holdout_rows),
            "latency": aggregate_latency([{"fallback": row} for row in holdout_rows], "fallback"),
            "existing_m5_baseline": existing_holdout_baseline(len(evaluation)),
        },
        "results": holdout_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "m9_iterative.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CodeAgent RAG.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cross-encoder", help="Local cross-encoder path or cached model name.")
    parser.add_argument("--reformulations", type=Path, help="JSON object mapping benchmark IDs to frozen reformulations.")
    parser.add_argument("--retry-threshold", type=float, help="Frozen top rerank-score threshold for one fallback.")
    parser.add_argument("--bm25-only", action="store_true", help="Run BM25 without loading an embedding model.")
    parser.add_argument("--local-reranker", action="store_true", help="Train on development data and evaluate a local pointwise reranker.")
    parser.add_argument("--iterative-bm25", action="store_true", help="Calibrate one BM25 reformulation on development data and evaluate holdout.")
    args = parser.parse_args()

    offline = args.bm25_only or args.local_reranker or args.iterative_bm25
    questions = load_questions(args.questions, require_sources=not offline)
    if args.bm25_only:
        run_bm25_only(questions, args.output_dir)
    if args.local_reranker:
        run_local_reranker(questions, args.output_dir)
    if args.iterative_bm25:
        run_iterative_bm25(questions, args.output_dir)
    if offline:
        return
    retriever, chunk_count = load_retriever()
    cross_encoder = None
    if args.cross_encoder:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(args.cross_encoder, local_files_only=True)
    reformulations = json.loads(args.reformulations.read_text(encoding="utf-8")) if args.reformulations else {}
    print(f"\nRunning {len(questions)} benchmark questions...\n")
    results = []

    for item in questions:
        result = evaluate_question(
            retriever,
            item,
            cross_encoder=cross_encoder,
            reformulated_question=reformulations.get(item["id"]),
            retry_threshold=args.retry_threshold,
        )
        results.append(result)
        print(
            f"{result['id']} | "
            f"M4 H@5={result['m4']['metrics']['hit@5']:.0f} | "
            f"M5 H@5={result['m5_reranking']['final']['metrics']['hit@5']:.0f} | "
            f"Context H@5={result['m5_context']['metrics']['hit@5']:.0f}"
        )

    dense_aggregate = aggregate_metrics(results, lambda result: result["dense"]["metrics"])
    lexical_aggregate = aggregate_metrics(results, lambda result: result["lexical"]["metrics"])
    bm25_aggregate = aggregate_metrics(results, lambda result: result["bm25"]["metrics"])
    hybrid_aggregate = aggregate_metrics(results, lambda result: result["hybrid"]["metrics"])
    m4_aggregate = aggregate_metrics(results, lambda result: result["m4"]["metrics"])
    m5_final_aggregate = aggregate_metrics(
        results,
        lambda result: result["m5_reranking"]["final"]["metrics"],
    )
    context_aggregate = aggregate_metrics(results, lambda result: result["m5_context"]["metrics"])
    reranking_strategies = aggregate_reranking_strategies(results)
    m4_diagnosis = aggregate_diagnosis(results, lambda result: result["m4"]["diagnosis"])
    m5_diagnosis = aggregate_diagnosis(results, lambda result: result["m5_reranking"]["diagnosis"])
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            "bm25": bm25_aggregate,
            "hybrid": hybrid_aggregate,
            "final": m5_final_aggregate,
            "reranking_strategies": reranking_strategies,
            "diagnosis": m5_diagnosis,
            "latency": {
                name: aggregate_latency(results, name)
                for name in ("dense", "lexical", "bm25", "hybrid", "specialized", "fallback")
            },
        },
        "results": [
            {
                "id": result["id"],
                "question": result["question"],
                "expected_sources": result["expected_sources"],
                "dense": result["dense"],
                "lexical": result["lexical"],
                "bm25": result["bm25"],
                "hybrid": result["hybrid"],
                "final": result["m5_reranking"]["final"],
                "reranking_strategies": result["m5_reranking"]["strategies"],
                "diagnosis": result["m5_reranking"]["diagnosis"],
                "specialized": result["specialized"],
                "fallback": result["fallback"],
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

    write_json(args.output_dir / "m4_baseline.json", m4_payload)
    write_json(args.output_dir / "m5_reranking.json", m5_reranking_payload)
    write_json(args.output_dir / "m5_context.json", m5_context_payload)


if __name__ == "__main__":
    main()
