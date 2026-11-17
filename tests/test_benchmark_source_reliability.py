from scripts.benchmark_source_reliability import run


def test_reliability_benchmark_counts_false_results():
    cases = [{
        "id": "case",
        "chunks": [{"case_source_id": "source", "file": "a.py", "obsolete": True}],
        "expected_stale": [],
        "expected_contradictions": [],
    }]

    result = run(cases)

    assert result["summary"]["stale"] == {
        "true_positive": 0,
        "false_positive": 1,
        "false_negative": 0,
    }
