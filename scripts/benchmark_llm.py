import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BENCHMARK_PATH = ROOT / "benchmarks" / "llm" / "benchmark.json"
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "llm" / "results"
DEFAULT_BASE_URL = "http://localhost:11434"


def load_benchmark(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def call_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    generation: dict,
    timeout: float,
) -> dict:
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": generation["temperature"],
            "seed": generation["seed"],
            "num_predict": generation["num_predict"],
        },
    }

    req = request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()

    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Unable to reach Ollama at {base_url}. Check that the local server is running."
        ) from exc

    data["_wall_time_seconds"] = time.perf_counter() - started_at
    return data


def build_prompt(test: dict) -> str:
    return f"CONTEXTE:\n{test['context']}\n\nQUESTION:\n{test['question']}"


def nanoseconds_to_seconds(value: int | None) -> float | None:
    if value is None:
        return None

    return value / 1_000_000_000


def compute_tokens_per_second(
    eval_count: int | None,
    eval_duration: int | None,
) -> float | None:
    if not eval_count or not eval_duration:
        return None

    duration_seconds = nanoseconds_to_seconds(eval_duration)

    if not duration_seconds:
        return None

    return eval_count / duration_seconds


def safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def warm_up(
    base_url: str,
    model: str,
    system_prompt: str,
    generation: dict,
    timeout: float,
) -> None:
    warmup_generation = dict(generation)
    warmup_generation["num_predict"] = 8

    print(f"Warm-up du modèle {model}...")

    call_ollama(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        prompt="Réponds uniquement par : OK",
        generation=warmup_generation,
        timeout=timeout,
    )


def run_test(
    test: dict,
    model: str,
    base_url: str,
    system_prompt: str,
    generation: dict,
    timeout: float,
) -> dict:
    response = call_ollama(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        prompt=build_prompt(test),
        generation=generation,
        timeout=timeout,
    )

    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    tokens_per_second = compute_tokens_per_second(eval_count, eval_duration)

    return {
        "id": test["id"],
        "category": test["category"],
        "question": test["question"],
        "response": response.get("response", "").strip(),
        "thinking": response.get("thinking"),
        "reference_answer": test["reference_answer"],
        "evaluation_notes": test["evaluation_notes"],
        "performance": {
            "wall_time_seconds": round(response["_wall_time_seconds"], 4),
            "total_duration_seconds": nanoseconds_to_seconds(response.get("total_duration")),
            "load_duration_seconds": nanoseconds_to_seconds(response.get("load_duration")),
            "prompt_tokens": response.get("prompt_eval_count"),
            "output_tokens": eval_count,
            "tokens_per_second": round(tokens_per_second, 2) if tokens_per_second is not None else None,
        },
    }


def build_summary(results: list[dict]) -> dict:
    latencies = [result["performance"]["wall_time_seconds"] for result in results]

    speeds = [
        result["performance"]["tokens_per_second"]
        for result in results
        if result["performance"]["tokens_per_second"] is not None
    ]

    total_output_tokens = sum(
        result["performance"]["output_tokens"] or 0
        for result in results
    )

    return {
        "tests": len(results),
        "total_output_tokens": total_output_tokens,
        "mean_latency_seconds": round(statistics.mean(latencies), 4),
        "median_latency_seconds": round(statistics.median(latencies), 4),
        "mean_tokens_per_second": round(statistics.mean(speeds), 2) if speeds else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark commun des LLM locaux pour CodeAgent."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Nom du modèle local à benchmarker.",
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Chemin vers benchmark.json.",
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="URL du serveur Ollama.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout maximal d'une requête en secondes.",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Désactive le warm-up initial du modèle.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = load_benchmark(args.benchmark)

    system_prompt = benchmark["system_prompt"]
    generation = benchmark["generation"]
    tests = benchmark["tests"]

    if not args.skip_warmup:
        warm_up(
            base_url=args.base_url,
            model=args.model,
            system_prompt=system_prompt,
            generation=generation,
            timeout=args.timeout,
        )

    results = []

    print()
    print(f"Modèle : {args.model}")
    print(f"Tests  : {len(tests)}")
    print()

    for index, test in enumerate(tests, start=1):
        print(f"[{index}/{len(tests)}] {test['id']} ({test['category']})")

        result = run_test(
            test=test,
            model=args.model,
            base_url=args.base_url,
            system_prompt=system_prompt,
            generation=generation,
            timeout=args.timeout,
        )

        results.append(result)

        latency = result["performance"]["wall_time_seconds"]
        speed = result["performance"]["tokens_per_second"]

        print(f"  Latence : {latency:.2f} s")

        if speed is not None:
            print(f"  Débit   : {speed:.2f} tokens/s")

    summary = build_summary(results)

    output = {
        "benchmark_version": benchmark["version"],
        "model": args.model,
        "runtime": "ollama",
        "base_url": args.base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation": generation,
        "quality_weights": benchmark["quality_weights"],
        "summary": summary,
        "results": results,
    }

    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DEFAULT_RESULTS_DIR / f"{safe_model_name(args.model)}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print()
    print("=== RÉSUMÉ ===")
    print(f"Tests           : {summary['tests']}")
    print(f"Latence moyenne : {summary['mean_latency_seconds']:.2f} s")
    print(f"Latence médiane : {summary['median_latency_seconds']:.2f} s")

    if summary["mean_tokens_per_second"] is not None:
        print(f"Débit moyen     : {summary['mean_tokens_per_second']:.2f} tokens/s")

    print(f"Tokens générés  : {summary['total_output_tokens']}")
    print(f"Résultats       : {output_path}")


if __name__ == "__main__":
    main()