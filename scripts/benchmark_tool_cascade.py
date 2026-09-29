import json
import statistics
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter

from rlcd import Choice, DecisionEngine, Option

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "benchmarks" / "tools" / "gating_holdout.json"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "tools" / "results"
RESULT_PATH = RESULTS_DIR / "verdict_devstral_cascade_holdout.json"

VERDICT_THRESHOLD = 0.45
DEVSTRAL_MODEL = "devstral:24b"
OLLAMA_URL = "http://localhost:11434/api/chat"

TOOL_CRITERIA = {
    "no_tool": "Aucun outil projet n'est nécessaire.",
    "list_files": "Lister les fichiers ou dossiers du projet.",
    "read_file": "Lire le contenu d'un fichier.",
    "search_code": "Rechercher du texte ou du code dans le projet.",
    "get_git_diff": "Inspecter les modifications Git.",
    "run_tests": "Exécuter les tests du projet.",
}

DEVSTRAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": TOOL_CRITERIA["list_files"],
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": TOOL_CRITERIA["read_file"],
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": ["integer", "null"]},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": TOOL_CRITERIA["search_code"],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_git_diff",
            "description": TOOL_CRITERIA["get_git_diff"],
            "parameters": {
                "type": "object",
                "properties": {"staged": {"type": "boolean"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": TOOL_CRITERIA["run_tests"],
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
            },
        },
    },
]


def expected_tools(task: dict) -> set[str]:
    if "expected_tools" in task:
        return set(task["expected_tools"])
    tool = task.get("expected_tool")
    return {tool} if tool is not None else set()


def prediction_is_correct(prediction: str | None, task: dict) -> bool:
    expected = expected_tools(task)
    if expected:
        return prediction in expected
    return prediction == "no_tool"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def build_verdict():
    engine = DecisionEngine()
    query = Choice(
        id="tool",
        question="Quel outil faut-il utiliser pour traiter la requête utilisateur ?",
        options=[
            Option(id=tool, description=description)
            for tool, description in TOOL_CRITERIA.items()
        ],
    )
    return engine, query


def verdict_predict(engine, query, prompt: str) -> dict:
    start = perf_counter()
    result = engine.evaluate(context=prompt, queries=[query])
    wall_latency = perf_counter() - start
    decision = result.results[0]
    return {
        "prediction": decision.selected_id,
        "confidence": float(decision.selected_probability),
        "is_abstention": bool(decision.is_abstention),
        "probabilities": {
            key: float(value)
            for key, value in decision.probabilities.items()
        },
        "internal_latency_seconds": float(decision.latency_ms) / 1000.0,
        "wall_latency_seconds": wall_latency,
    }


def devstral_predict(prompt: str) -> dict:
    payload = {
        "model": DEVSTRAL_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "tools": DEVSTRAL_TOOLS,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 512,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {body}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError("Unable to reach Ollama or request timed out.") from error
    latency = perf_counter() - start
    message = data.get("message", {})
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        prediction = "no_tool"
    else:
        prediction = (
            tool_calls[0].get("function", {}).get("name")
        )
    return {"prediction": prediction, "latency_seconds": latency}


def warm_up(engine, query) -> None:
    print("Warming up Verdict...")
    verdict_predict(
        engine,
        query,
        "Réponds uniquement par OK.",
    )
    print("Warming up Devstral...")
    devstral_predict("Réponds uniquement par OK.")
    print("Warm-up complete.\n")


def run_task(engine, query, task: dict) -> dict:
    error = None
    verdict = None
    fallback = None
    cascade_prediction = None
    cascade_latency = 0.0
    try:
        verdict = verdict_predict(
            engine,
            query,
            task["prompt"],
        )
        should_fallback = (
            verdict["is_abstention"]
            or verdict["confidence"] < VERDICT_THRESHOLD
            or verdict["prediction"] == "__insufficient_evidence__"
        )
        if should_fallback:
            fallback = devstral_predict(task["prompt"])
            cascade_prediction = fallback["prediction"]
            cascade_latency = (
                verdict["wall_latency_seconds"]
                + fallback["latency_seconds"]
            )
        else:
            cascade_prediction = verdict["prediction"]
            cascade_latency = verdict["wall_latency_seconds"]
        devstral = devstral_predict(task["prompt"])
    except Exception as exc:
        error = str(exc)
        devstral = None
    return {
        "id": task["id"],
        "prompt": task["prompt"],
        "expected_tools": sorted(expected_tools(task)),
        "verdict_prediction": (
            verdict["prediction"]
            if verdict
            else None
        ),
        "verdict_confidence": (
            verdict["confidence"]
            if verdict
            else None
        ),
        "verdict_abstention": (
            verdict["is_abstention"]
            if verdict
            else None
        ),
        "verdict_correct": (
            prediction_is_correct(verdict["prediction"], task)
            if verdict
            else False
        ),
        "fallback_used": fallback is not None,
        "fallback_prediction": (
            fallback["prediction"]
            if fallback
            else None
        ),
        "cascade_prediction": cascade_prediction,
        "cascade_correct": (
            prediction_is_correct(cascade_prediction, task)
            if cascade_prediction is not None
            else False
        ),
        "cascade_latency_seconds": round(cascade_latency, 4),
        "devstral_prediction": (
            devstral["prediction"]
            if devstral
            else None
        ),
        "devstral_correct": (
            prediction_is_correct(devstral["prediction"], task)
            if devstral
            else False
        ),
        "devstral_latency_seconds": (
            round(devstral["latency_seconds"], 4)
            if devstral
            else None
        ),
        "verdict_latency_seconds": (
            round(verdict["wall_latency_seconds"], 4)
            if verdict
            else None
        ),
        "verdict_internal_latency_seconds": (
            round(verdict["internal_latency_seconds"], 4)
            if verdict
            else None
        ),
        "error": error,
    }


def summarize(results: list[dict]) -> dict:
    valid = [
        result
        for result in results
        if result["error"] is None
    ]
    cascade_latencies = [
        result["cascade_latency_seconds"]
        for result in valid
    ]
    devstral_latencies = [
        result["devstral_latency_seconds"]
        for result in valid
    ]
    verdict_latencies = [
        result["verdict_latency_seconds"]
        for result in valid
    ]
    fallback_count = sum(result["fallback_used"] for result in valid)
    direct_count = len(valid) - fallback_count
    direct_results = [
        result
        for result in valid
        if not result["fallback_used"]
    ]
    cascade_accuracy = (
        sum(result["cascade_correct"] for result in valid)
        / len(valid)
        if valid
        else 0.0
    )
    devstral_accuracy = (
        sum(result["devstral_correct"] for result in valid)
        / len(valid)
        if valid
        else 0.0
    )
    verdict_raw_accuracy = (
        sum(result["verdict_correct"] for result in valid)
        / len(valid)
        if valid
        else 0.0
    )
    verdict_direct_accuracy = (
        sum(result["cascade_correct"] for result in direct_results)
        / len(direct_results)
        if direct_results
        else 0.0
    )
    average_cascade_latency = (
        sum(cascade_latencies)
        / len(cascade_latencies)
        if cascade_latencies
        else 0.0
    )
    average_devstral_latency = (
        sum(devstral_latencies)
        / len(devstral_latencies)
        if devstral_latencies
        else 0.0
    )
    latency_reduction = (
        1.0
        - (
            average_cascade_latency
            / average_devstral_latency
        )
        if average_devstral_latency
        else 0.0
    )
    return {
        "tasks": len(results),
        "valid_tasks": len(valid),
        "threshold": VERDICT_THRESHOLD,
        "cascade_accuracy": cascade_accuracy,
        "devstral_accuracy": devstral_accuracy,
        "verdict_raw_accuracy": verdict_raw_accuracy,
        "verdict_direct_accuracy": verdict_direct_accuracy,
        "verdict_direct_tasks": direct_count,
        "verdict_direct_rate": (
            direct_count / len(valid)
            if valid
            else 0.0
        ),
        "fallback_tasks": fallback_count,
        "fallback_rate": (
            fallback_count / len(valid)
            if valid
            else 0.0
        ),
        "average_cascade_latency_seconds": average_cascade_latency,
        "median_cascade_latency_seconds": (
            statistics.median(cascade_latencies)
            if cascade_latencies
            else 0.0
        ),
        "p95_cascade_latency_seconds": percentile(cascade_latencies, 0.95),
        "average_devstral_latency_seconds": average_devstral_latency,
        "median_devstral_latency_seconds": (
            statistics.median(devstral_latencies)
            if devstral_latencies
            else 0.0
        ),
        "p95_devstral_latency_seconds": percentile(devstral_latencies, 0.95),
        "average_verdict_latency_seconds": (
            sum(verdict_latencies)
            / len(verdict_latencies)
            if verdict_latencies
            else 0.0
        ),
        "latency_reduction_fraction": latency_reduction,
        "latency_reduction_percent": latency_reduction * 100.0,
        "errors": sum(result["error"] is not None for result in results),
    }


def main():
    tasks = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )
    engine, query = build_verdict()
    warm_up(engine, query)
    results = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] " f"{task['id']}")
        result = run_task(
            engine,
            query,
            task,
        )
        results.append(result)
        print(
            f"  cascade={result['cascade_correct']} "
            f"prediction={result['cascade_prediction']} "
            f"fallback={result['fallback_used']} "
            f"confidence={result['verdict_confidence']} "
            f"cascade_latency="
            f"{result['cascade_latency_seconds']:.3f}s"
        )
        if result["error"]:
            print(f"  error={result['error']}")
    summary = summarize(results)
    output = {
        "benchmark": "verdict_devstral_cascade_holdout",
        "dataset": DATASET_PATH.name,
        "verdict_threshold": VERDICT_THRESHOLD,
        "verdict_model": "Verdict",
        "fallback_model": DEVSTRAL_MODEL,
        "summary": summary,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\nSummary")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
