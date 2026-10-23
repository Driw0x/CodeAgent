import json

import pytest

import scripts.benchmark_planning as benchmark_planning

from scripts.benchmark_planning import FixtureLLM, run, scenarios, validated_steps


@pytest.mark.anyio
async def test_planning_benchmark_replans_without_repeating_completed_work():
    payload = await run("replan")
    assert all(payload["checks"].values())
    direct, planned = payload["results"]
    assert direct["success"] and planned["success"]
    assert planned["replans"] == 1
    assert planned["repeated_calls"] == 0
    assert planned["actual_model_tokens"] is None
    assert planned["fixture_token_estimates"]["total"] > 0
    preserved = next(event for event in planned["execution_state"]["events"] if event["event"] == "replanning")
    assert preserved["preserved"] == ["1"]
    assert planned["execution_state"]["plan"][0]["evidence"] == ["S1"]
    assert len(planned["step_seconds"]["2"]) == 2


def test_benchmark_evidence_checks_ignore_metadata_and_empty_results():
    execution = {"plan": [{"id": "1", "status": "COMPLETED", "depends_on": [], "evidence": ["S1"],
                           "success": {"tool": "search_code", "kind": "contains", "value": "marker"}}]}
    observations = {"S1": {"tool": "search_code", "result": {"query": "marker", "matches": []}}}
    assert validated_steps(execution, observations) == [False]
    grounder = FixtureLLM(scenarios()[0])
    assert grounder.generate("SOURCES: return 0\nAFFIRMATION À VÉRIFIER:\nmarker retourne 42") == "UNSUPPORTED"


@pytest.mark.anyio
async def test_benchmark_reports_actual_mcp_calls_separately_from_blocked_attempts():
    payload = await run("tool_budget", execution="auto")
    row = payload["results"][0]
    assert row["tool_calls"] == 1
    assert row["tool_attempts"] == 2
    assert not row["success"]
    assert row["execution_state"]["remaining"] == ["2"]


@pytest.mark.anyio
async def test_real_benchmark_preserves_report_when_usage_is_unavailable(monkeypatch, capsys):
    fixture = await run("simple", execution="auto")
    template = fixture["results"][0]

    async def without_usage(case, execution, *args):
        return {**template, "case": case["id"], "execution": execution, "simulated": False,
                "success": False, "expectation_met": False, "actual_model_tokens": None,
                "fixture_token_estimates": None, "error": "Ollama request timed out before usage was reported."}

    monkeypatch.setattr(benchmark_planning, "evaluate", without_usage)
    payload = await run("simple", execution="auto", real=True)
    report = json.loads(json.dumps(payload))
    assert report["results"][0]["error"] == "Ollama request timed out before usage was reported."
    assert report["summary"]["auto"]["total_actual_model_tokens"] is None
    assert report["summary"]["auto"]["total_fixture_token_estimates"] is None
    benchmark_planning.print_comparison(report)
    assert "unavailable" in capsys.readouterr().out


def test_R9_benchmark_independent_oracle_uses_decoded_text():
    from scripts.benchmark_planning import validated_steps
    condition = {"tool": "read_file", "kind": "contains", "value": 'return "hello"'}
    execution = {"plan": [{"id": "1", "status": "COMPLETED", "success": condition, "evidence": ["S1"], "depends_on": []}]}
    observations = {"S1": {"tool": "read_file", "result": {"content": 'return "hello"'}}}
    assert validated_steps(execution, observations) == [True]
    observations["S1"]["result"] = {"content": "other", "path": 'return "hello"'}
    assert validated_steps(execution, observations) == [False]
