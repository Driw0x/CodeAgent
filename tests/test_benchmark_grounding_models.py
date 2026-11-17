from scripts.benchmark_grounding_models import (
    calibrate_independent_verifier,
    independent_verifier_score,
    summarize_rows,
)


def test_grounding_summary_reports_confusion_and_percentiles():
    cases = [{"id": str(index)} for index in range(4)]
    rows = [
        {"expected": True, "predicted": True, "correct": True, "latency_seconds": 1.0},
        {"expected": True, "predicted": False, "correct": False, "latency_seconds": 2.0},
        {"expected": False, "predicted": True, "correct": False, "latency_seconds": 3.0},
        {"expected": False, "predicted": False, "correct": True, "latency_seconds": 4.0},
    ]

    summary = summarize_rows(
        "candidate", cases, rows, candidate_type="nli", cold_load_seconds=0.5, benchmark_seconds=10.0
    )

    assert summary["false_accepts"] == 1
    assert summary["false_rejects"] == 1
    assert summary["confusion_matrix"] == {
        "true_accepts": 1,
        "false_accepts": 1,
        "true_rejects": 1,
        "false_rejects": 1,
    }
    assert summary["p50_latency_seconds"] == 2.5
    assert summary["p95_latency_seconds"] == 3.8499999999999996


def test_independent_verifier_calibrates_without_holdout_labels():
    supported = {"answer": "value 42 [S1]", "sources": [{"content": "value = 42"}], "expected": True}
    unsupported = {"answer": "value 99 [S1]", "sources": [{"content": "value = 42"}], "expected": False}

    threshold = calibrate_independent_verifier([supported, unsupported])

    assert independent_verifier_score(supported["answer"], supported["sources"]) >= threshold
    assert independent_verifier_score(unsupported["answer"], unsupported["sources"]) < threshold
