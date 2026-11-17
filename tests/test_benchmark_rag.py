import pytest

from scripts.benchmark_rag import (
    aggregate_latency,
    calculate_metrics,
    expanded_query,
    local_rerank,
    relative_path,
    source_matches,
    specialized_rerank,
)


EXPECTED = [{"file": "a.py", "type": "function", "name": "target"}]


def test_rag_metrics_include_ndcg():
    retrieved = [
        {"file": "x.py", "type": "function", "name": "other"},
        {"file": "a.py", "type": "function", "name": "target"},
    ]

    metrics = calculate_metrics(retrieved, EXPECTED, (1, 2))

    assert metrics["ndcg@1"] == 0.0
    assert metrics["ndcg@2"] == pytest.approx(1 / 1.5849625007)


def test_latency_summary_reports_p50_and_p95():
    results = [{"bm25": {"latency_seconds": value}} for value in (1.0, 2.0, 10.0)]

    summary = aggregate_latency(results, "bm25")

    assert summary == {"mean_seconds": pytest.approx(13 / 3), "p50_seconds": 2.0, "p95_seconds": pytest.approx(9.2)}


def test_specialized_reranker_uses_model_scores():
    class Model:
        def predict(self, pairs, show_progress_bar=False):
            assert len(pairs) == 2
            return [0.1, 0.9]

    chunks = [
        {"file": "a.py", "name": "first", "content": "one"},
        {"file": "b.py", "name": "second", "content": "two"},
    ]

    assert specialized_rerank(Model(), "question", chunks)[0]["name"] == "second"


def test_frozen_fixture_paths_are_portable_between_workspaces():
    chunk = {"file": "E:/old/workspace/CodeAgent/app/parser/file_loader.py", "type": "function", "name": "read"}

    assert relative_path(chunk["file"]) == "app/parser/file_loader.py"
    assert relative_path(r"E:\old\workspace\CodeAgent\app\parser\file_loader.py") == "app/parser/file_loader.py"
    assert source_matches(chunk, EXPECTED[0]) is False
    assert source_matches(chunk, {"file": "app/parser/file_loader.py", "type": "function", "name": "read"}) is True


def test_local_reranker_and_expansion_are_deterministic():
    class Model:
        def predict_proba(self, features):
            return __import__("numpy").array([[0.9, 0.1], [0.1, 0.9]])

    chunks = [
        {"file": "a.py", "name": "first", "type": "function", "content": "one", "bm25_score": 1.0},
        {"file": "b.py", "name": "second", "type": "function", "content": "two", "bm25_score": 2.0},
    ]

    assert local_rerank(Model(), "question", chunks)[0]["name"] == "second"
    assert expanded_query("question", chunks) == "question first second"
