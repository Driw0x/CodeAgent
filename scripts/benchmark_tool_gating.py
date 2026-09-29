import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "benchmarks" / "tools" / "baseline_tasks.json"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "tools" / "results"

TOOL_CRITERIA = {
    "no_tool": "Aucun outil projet n'est nécessaire.",
    "list_files": "Lister les fichiers ou dossiers du projet.",
    "read_file": "Lire le contenu d'un fichier.",
    "search_code": "Rechercher du texte ou du code dans le projet.",
    "get_git_diff": "Inspecter les modifications Git.",
    "run_tests": "Exécuter les tests du projet.",
}

TOOL_IDS = list(TOOL_CRITERIA)

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

Laya_QUESTIONS = {
    "tool": {
        "type": "choice",
        "instructions": "Choisis l'outil le plus approprié pour traiter la requête utilisateur.",
        "criteria": TOOL_CRITERIA,
    }
}


def expected_tools(task: dict) -> set[str]:
    if "expected_tools" in task:
        return set(task["expected_tools"])
    tool = task.get("expected_tool")
    return {tool} if tool is not None else set()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def find_choice(value):
    if isinstance(value, dict):
        for key in ("choice", "selected_option_id", "selected", "label"):
            selected = value.get(key)
            if selected in TOOL_IDS:
                confidence = value.get("confidence")
                if confidence is None:
                    confidence = value.get("answer_confidence")
                if confidence is not None:
                    confidence = float(confidence)
                return selected, confidence

        for child in value.values():
            result = find_choice(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find_choice(child)
            if result is not None:
                return result

    return None


def build_devstral():
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.llm import LocalLLM
    llm = LocalLLM(model="devstral:24b")

    def predict(prompt: str):
        message = llm.chat(messages=[{"role": "user", "content": prompt}], tools=DEVSTRAL_TOOLS)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return "no_tool", None, {"content": message.get("content", "")}
        function = tool_calls[0].get("function", {})
        return function.get("name"), None, {"arguments": function.get("arguments", {})}

    def warm_up():
        predict("Réponds uniquement par OK.")

    return predict, warm_up


def build_laya():
    from laya import Router
    router = Router()

    def predict(prompt: str):
        result = router.predict(prompt, Laya_QUESTIONS)
        answer = result["answers"]["tool"]
        return (
            answer["choice"],
            float(answer.get("confidence", answer.get("answer_confidence", 0.0))),
            {
                "probabilities": answer.get("probabilities"),
                "routing": result.get("routing"),
            },
        )

    def warm_up():
        predict("Réponds uniquement par OK.")

    return predict, warm_up


def build_gliner():
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")

    def predict(prompt: str):
        result = model.classify_text(prompt, {"tool": TOOL_IDS}, include_confidence=True)
        answer = result["tool"]
        if isinstance(answer, dict):
            return answer["label"], float(answer.get("confidence", 0.0)), {}
        return answer, None, {}

    def warm_up():
        predict("Réponds uniquement par OK.")

    return predict, warm_up


def build_modernbert():
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model="tasksource/ModernBERT-base-nli")

    def predict(prompt: str):
        result = classifier(prompt, TOOL_IDS, multi_label=False)
        return (
            result["labels"][0],
            float(result["scores"][0]),
            {
                "labels": result["labels"],
                "scores": [
                    float(score)
                    for score in result["scores"]
                ],
            },
        )

    def warm_up():
        predict("Réponds uniquement par OK.")
    return predict, warm_up


def build_verdict():
    from rlcd import Choice, DecisionEngine, Option
    engine = DecisionEngine()
    query = Choice(
        id="tool",
        question="Quel outil faut-il utiliser pour traiter la requête utilisateur ?",
        options=[
            Option(id=tool, description=description)
            for tool, description in TOOL_CRITERIA.items()
        ],
    )

    def predict(prompt: str):
        result = engine.evaluate(context=prompt, queries=[query])
        decision = result.results[0]
        return (
            decision.selected_id,
            float(decision.selected_probability),
            {
                "probabilities": {
                    key: float(value)
                    for key, value in decision.probabilities.items()
                },
                "is_abstention": decision.is_abstention,
                "model_id": decision.model_id,
                "calibration_status": decision.calibration_status,
                "latency_ms": float(decision.latency_ms),
            },
        )

    def warm_up():
        predict("Réponds uniquement par OK.")
    return predict, warm_up


def build_jev():
    key = os.getenv("JEV_API_KEY")

    if not key:
        raise RuntimeError("JEV_API_KEY is not defined.")

    def predict(prompt: str):
        payload = {
            "state": {"request": prompt},
            "questions": {
                "tool": {
                    "type": "choice",
                    "instructions": "Choisis l'outil le plus approprié pour traiter la requête utilisateur.",
                    "criteria": TOOL_CRITERIA,
                }
            },
        }
        request = urllib.request.Request(
            "https://www.jevai.org/api/v1/decisions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Jev HTTP {error.code}: {body}") from error

        found = find_choice(result)

        if found is None:
            raise RuntimeError(f"Unable to parse Jev choice: {result}")

        choice, confidence = found

        return choice, confidence, {"code": result.get("code"), "message": result.get("message")}

    def warm_up():
        pass

    return predict, warm_up


def build_router(name: str):
    builders = {
        "devstral": build_devstral,
        "laya": build_laya,
        "gliner": build_gliner,
        "modernbert": build_modernbert,
        "verdict": build_verdict,
        "jev": build_jev,
    }
    return builders[name]()


def run_task(predict, task: dict) -> dict:
    valid_tools = expected_tools(task)
    start = perf_counter()
    error = None
    predicted_tool = None
    confidence = None
    details = {}

    try:
        predicted_tool, confidence, details = predict(task["prompt"])
    except Exception as exc:
        error = str(exc)

    latency = perf_counter() - start

    if valid_tools:
        correct = predicted_tool in valid_tools
    else:
        correct = predicted_tool == "no_tool"

    return {
        "id": task["id"],
        "prompt": task["prompt"],
        "expected_tools": sorted(valid_tools),
        "predicted_tool": predicted_tool,
        "correct": correct and error is None,
        "confidence": confidence,
        "latency_seconds": round(latency, 4),
        "error": error,
        "details": details,
    }


def summarize(results: list[dict]) -> dict:
    tool_tasks = [result for result in results if result["expected_tools"]]
    no_tool_tasks = [result for result in results if not result["expected_tools"]]
    successful = [result for result in results if result["error"] is None]
    latencies = [result["latency_seconds"] for result in successful]
    confidences = [result["confidence"] for result in successful if result["confidence"] is not None]

    return {
        "tasks": len(results),
        "gating_accuracy": (
            sum(result["correct"] for result in results)
            / len(results)
        ),
        "tool_selection_accuracy": (
            sum(result["correct"] for result in tool_tasks)
            / len(tool_tasks)
            if tool_tasks
            else 0.0
        ),
        "no_tool_accuracy": (
            sum(result["correct"] for result in no_tool_tasks)
            / len(no_tool_tasks)
            if no_tool_tasks
            else 0.0
        ),
        "average_latency_seconds": (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        ),
        "median_latency_seconds": (
            statistics.median(latencies)
            if latencies
            else 0.0
        ),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "average_confidence": (
            sum(confidences) / len(confidences)
            if confidences
            else None
        ),
        "errors": sum(result["error"] is not None for result in results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--router",
        required=True,
        choices=[
            "devstral",
            "laya",
            "gliner",
            "modernbert",
            "verdict",
            "jev",
        ],
    )
    args = parser.parse_args()
    tasks = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"Loading {args.router}...")
    predict, warm_up = build_router(args.router)
    print(f"Warming up {args.router}...")
    warm_up()
    print("Warm-up complete.\n")
    results = []

    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task['id']}")
        result = run_task(predict, task)
        results.append(result)
        print(
            f"  correct={result['correct']} "
            f"predicted={result['predicted_tool']} "
            f"confidence={result['confidence']} "
            f"latency={result['latency_seconds']:.4f}s"
        )
        if result["error"]:
            print(f"  error={result['error']}")

    summary = summarize(results)
    output = {
        "router": args.router,
        "dataset": DATASET_PATH.name,
        "benchmark": "tool_gating",
        "summary": summary,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{args.router}_gating.json"
    result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {result_path}")


if __name__ == "__main__":
    main()
