import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.rag.reliability import prepare_reliable_sources

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "benchmarks" / "rag" / "reliability_cases.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "rag" / "results" / "m9_reliability.json"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def scores(expected: set[str], actual: set[str]) -> dict:
    true_positive = len(expected & actual)
    false_positive = len(actual - expected)
    false_negative = len(expected - actual)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def run(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        start = perf_counter()
        current, contradictions = prepare_reliable_sources(case["chunks"], case.get("current_versions"))
        latency = perf_counter() - start
        all_ids = {chunk["case_source_id"] for chunk in case["chunks"]}
        current_ids = {chunk["case_source_id"] for chunk in current}
        stale = all_ids - current_ids
        rows.append({
            "id": case["id"],
            "expected_stale": case["expected_stale"],
            "actual_stale": sorted(stale),
            "expected_contradictions": case["expected_contradictions"],
            "actual_contradictions": sorted(contradictions),
            "stale_scores": scores(set(case["expected_stale"]), stale),
            "contradiction_scores": scores(set(case["expected_contradictions"]), contradictions),
            "latency_seconds": latency,
        })
    latencies = [row["latency_seconds"] for row in rows]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(rows),
        "summary": {
            "stale": {
                key: sum(row["stale_scores"][key] for row in rows)
                for key in ("true_positive", "false_positive", "false_negative")
            },
            "contradictions": {
                key: sum(row["contradiction_scores"][key] for row in rows)
                for key in ("true_positive", "false_positive", "false_negative")
            },
            "p50_latency_seconds": percentile(latencies, 0.50),
            "p95_latency_seconds": percentile(latencies, 0.95),
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark stale and contradictory source detection.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    result = run(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
