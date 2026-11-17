import argparse
import json
import re
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from app.llm import LocalLLM
from app.rag.grounding import verify_grounding

DEFAULT_MODELS = ["qwen2.5-coder:14b", "devstral:24b"]
DEFAULT_DATASET = Path("benchmarks/grounding/grounding_holdout.json")
DEFAULT_RESULTS_DIR = Path("benchmarks/grounding/results")
DEFAULT_DEVELOPMENT_DATASET = Path("benchmarks/grounding/grounding_development.json")
LEXICAL_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "en", "est", "et",
    "la", "le", "les", "par", "pour", "que", "qui", "sont", "un", "une", "the", "is", "of", "to",
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Grounding dataset must be a non-empty JSON list.")
    required = {"id", "answer", "sources", "expected"}
    for index, case in enumerate(data):
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError(f"Invalid case at index {index}: required fields are {sorted(required)}.")
        if not isinstance(case["sources"], list) or not case["sources"]:
            raise ValueError(f"Case {case['id']} must contain at least one source.")
        if not isinstance(case["expected"], bool):
            raise ValueError(f"Case {case['id']} expected must be a boolean.")
    return data


def summarize_rows(
    label: str,
    cases: list[dict],
    rows: list[dict],
    *,
    candidate_type: str,
    cold_load_seconds: float,
    benchmark_seconds: float,
    ollama_load_seconds: float = 0.0,
    unload_error: str | None = None,
) -> dict:
    valid = [row for row in rows if row["predicted"] is not None]
    latencies = [row["latency_seconds"] for row in valid]
    positives = [row for row in valid if row["expected"]]
    negatives = [row for row in valid if not row["expected"]]
    true_positives = sum(row["predicted"] is True for row in positives)
    true_negatives = sum(row["predicted"] is False for row in negatives)
    false_accepts = sum(row["expected"] is False and row["predicted"] is True for row in valid)
    false_rejects = sum(row["expected"] is True and row["predicted"] is False for row in valid)
    prompt_tokens = [row.get("prompt_tokens", 0) for row in valid]
    completion_tokens = [row.get("completion_tokens", 0) for row in valid]
    total_tokens = [row.get("total_tokens", 0) for row in valid]
    return {
        "candidate": label,
        "model": label,
        "candidate_type": candidate_type,
        "cases": len(cases),
        "valid_cases": len(valid),
        "errors": len(cases) - len(valid),
        "accuracy": sum(row["correct"] for row in valid) / len(valid) if valid else 0.0,
        "supported_recall": true_positives / len(positives) if positives else 0.0,
        "unsupported_recall": true_negatives / len(negatives) if negatives else 0.0,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "confusion_matrix": {
            "true_accepts": true_positives,
            "false_accepts": false_accepts,
            "true_rejects": true_negatives,
            "false_rejects": false_rejects,
        },
        "cold_load_wall_seconds": cold_load_seconds,
        "ollama_load_duration_seconds": ollama_load_seconds,
        "benchmark_seconds": benchmark_seconds,
        "mean_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "mean_prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else 0.0,
        "mean_completion_tokens": statistics.mean(completion_tokens) if completion_tokens else 0.0,
        "mean_total_tokens": statistics.mean(total_tokens) if total_tokens else 0.0,
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "total_tokens": sum(total_tokens),
        "initial_unload_error": unload_error,
    }


def benchmark_model(model: str, cases: list[dict], keep_alive: str) -> dict:
    llm = LocalLLM(model=model, keep_alive=keep_alive)
    unload_error = None

    try:
        llm.unload()
    except RuntimeError as exc:
        unload_error = str(exc)

    cold_start = time.perf_counter()
    preload_stats = llm.preload()
    cold_load_wall_seconds = time.perf_counter() - cold_start
    rows = []
    total_start = time.perf_counter()

    for case in cases:
        llm.reset_usage_stats()
        start = time.perf_counter()
        error = None

        try:
            predicted = verify_grounding(case["answer"], case["sources"], llm)
        except Exception as exc:
            predicted = None
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - start
        usage = dict(llm.usage_stats)
        expected = case["expected"]
        correct = predicted is expected if predicted is not None else False
        rows.append({
            "id": case["id"],
            "category": case.get("category"),
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "latency_seconds": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "error": error,
        })

    total_seconds = time.perf_counter() - total_start
    summary = summarize_rows(
        model,
        cases,
        rows,
        candidate_type="generative_grounder",
        cold_load_seconds=cold_load_wall_seconds,
        benchmark_seconds=total_seconds,
        ollama_load_seconds=preload_stats.get("load_duration_ms", 0.0) / 1000,
        unload_error=unload_error,
    )

    try:
        llm.unload()
    except RuntimeError:
        pass

    return {"summary": summary, "cases": rows}


def benchmark_nli(model_name: str, cases: list[dict]) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    cold_load_seconds = time.perf_counter() - load_start
    entailment_ids = {
        int(index) for index, label in model.config.id2label.items()
        if "entail" in str(label).casefold()
    }


def lexical_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return {
        token for token in re.findall(r"[a-z0-9_.-]+", normalized)
        if token not in LEXICAL_STOPWORDS and not re.fullmatch(r"s\d+", token)
    }


def independent_verifier_score(answer: str, sources: list[dict]) -> float:
    claim = lexical_tokens(re.sub(r"\[S\d+\]", "", answer))
    evidence = lexical_tokens("\n".join(str(source.get("content", "")) for source in sources))
    return len(claim & evidence) / len(claim) if claim else 0.0


def calibrate_independent_verifier(cases: list[dict]) -> float:
    scored = [(independent_verifier_score(case["answer"], case["sources"]), case["expected"]) for case in cases]
    thresholds = sorted({0.0, 1.0000001, *(score for score, _ in scored)})
    candidates = []
    for threshold in thresholds:
        rows = [(score >= threshold, expected) for score, expected in scored]
        accuracy = sum(predicted is expected for predicted, expected in rows)
        false_accepts = sum(not expected and predicted for predicted, expected in rows)
        false_rejects = sum(expected and not predicted for predicted, expected in rows)
        candidates.append((accuracy, -false_accepts, -false_rejects, threshold))
    return max(candidates)[-1]


def benchmark_independent_verifier(cases: list[dict], development: list[dict]) -> dict:
    threshold = calibrate_independent_verifier(development)
    rows = []
    total_start = time.perf_counter()
    for case in cases:
        start = time.perf_counter()
        score = independent_verifier_score(case["answer"], case["sources"])
        predicted = score >= threshold
        expected = case["expected"]
        rows.append({
            "id": case["id"],
            "category": case.get("category"),
            "expected": expected,
            "predicted": predicted,
            "score": score,
            "threshold": threshold,
            "correct": predicted is expected,
            "latency_seconds": time.perf_counter() - start,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": None,
        })
    result = {
        "summary": summarize_rows(
            "independent_lexical",
            cases,
            rows,
            candidate_type="independent_lexical",
            cold_load_seconds=0.0,
            benchmark_seconds=time.perf_counter() - total_start,
        ),
        "calibration": {
            "dataset": str(DEFAULT_DEVELOPMENT_DATASET),
            "cases": len(development),
            "threshold": threshold,
        },
        "cases": rows,
    }
    return result
    if not entailment_ids:
        raise ValueError(f"{model_name}: no entailment label in model config.")
    rows = []
    total_start = time.perf_counter()
    for case in cases:
        start = time.perf_counter()
        error = None
        predicted = None
        try:
            premise = "\n\n".join(str(source.get("content", "")) for source in case["sources"])
            hypothesis = re.sub(r"\[S\d+\]", "", case["answer"]).strip()
            encoded = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
            with torch.no_grad():
                label_id = int(model(**encoded).logits.argmax(dim=-1).item())
            predicted = label_id in entailment_ids
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        expected = case["expected"]
        rows.append({
            "id": case["id"],
            "category": case.get("category"),
            "expected": expected,
            "predicted": predicted,
            "correct": predicted is expected if predicted is not None else False,
            "latency_seconds": time.perf_counter() - start,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": error,
        })
    total_seconds = time.perf_counter() - total_start
    return {
        "summary": summarize_rows(
            model_name,
            cases,
            rows,
            candidate_type="nli",
            cold_load_seconds=cold_load_seconds,
            benchmark_seconds=total_seconds,
        ),
        "cases": rows,
    }


def print_summary(result: dict):
    s = result["summary"]
    print(f"\nModel: {s['model']}")
    print(f"Cases                 : {s['cases']}")
    print(f"Valid cases           : {s['valid_cases']}")
    print(f"Errors                : {s['errors']}")
    print(f"Accuracy              : {s['accuracy']:.4f}")
    print(f"Supported recall      : {s['supported_recall']:.4f}")
    print(f"Unsupported recall    : {s['unsupported_recall']:.4f}")
    print(f"False accepts         : {s['false_accepts']}")
    print(f"False rejects         : {s['false_rejects']}")
    print(f"Cold load wall        : {s['cold_load_wall_seconds']:.2f} s")
    print(f"Ollama load duration  : {s['ollama_load_duration_seconds']:.2f} s")
    print(f"Mean warm latency     : {s['mean_latency_seconds']:.2f} s")
    print(f"Median warm latency   : {s['median_latency_seconds']:.2f} s")
    print(f"P50 warm latency      : {s['p50_latency_seconds']:.2f} s")
    print(f"P95 warm latency      : {s['p95_latency_seconds']:.2f} s")
    print(f"Mean prompt tokens    : {s['mean_prompt_tokens']:.1f}")
    print(f"Mean completion tokens: {s['mean_completion_tokens']:.1f}")
    print(f"Mean total tokens     : {s['mean_total_tokens']:.1f}")
    print(f"Total benchmark time  : {s['benchmark_seconds']:.2f} s")


def main():
    parser = argparse.ArgumentParser(description="Compare local models as CodeAgent grounding verifiers.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--nli-models", nargs="*", default=[], help="Local paths or cached Hugging Face NLI models.")
    parser.add_argument("--independent-lexical", action="store_true", help="Evaluate the calibrated local lexical verifier.")
    parser.add_argument("--development-dataset", type=Path, default=DEFAULT_DEVELOPMENT_DATASET)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = load_dataset(args.dataset)
    positives = sum(case["expected"] for case in cases)
    negatives = len(cases) - positives
    print(f"Dataset    : {args.dataset}")
    print(f"Cases      : {len(cases)}")
    print(f"Supported  : {positives}")
    print(f"Unsupported: {negatives}")
    results = []

    for model in args.models:
        print(f"\nBenchmarking {model}...")
        result = benchmark_model(model, cases, args.keep_alive)
        results.append(result)
        print_summary(result)

    for model in args.nli_models:
        print(f"\nBenchmarking local NLI {model}...")
        result = benchmark_nli(model, cases)
        results.append(result)
        print_summary(result)

    if args.independent_lexical:
        development = load_dataset(args.development_dataset)
        print("\nBenchmarking calibrated independent lexical verifier...")
        result = benchmark_independent_verifier(cases, development)
        results.append(result)
        print_summary(result)

    output = args.output

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = DEFAULT_RESULTS_DIR / f"grounding_models_{timestamp}.json"

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "models": args.models,
        "nli_models": args.nli_models,
        "independent_lexical": args.independent_lexical,
        "keep_alive": args.keep_alive,
        "generated_at": datetime.now().astimezone().isoformat(),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
