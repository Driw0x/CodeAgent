"""Adaptive decisions use M6 transport and M8 evidence/budgets, not model status."""

import json

import pytest

from app.agent.planning import adaptive_requirements, answer_code_observed
from app.agent.tool_loop import ToolLoop, build_tool_source
from app.memory.conversation import ConversationMemory
from app.mcp_server import mcp
from app.tools import project_tools
from tests.test_planning_integration import call, make_agent, read_result, search_result
from tests.test_tool_loop import FakeLLM, FakeGroundingLLM, tool_result


GOAL = "Localise marker, puis lis son implémentation et explique le résultat avec ses sources."
CODE = "def marker():\n    return 42\n"


@pytest.mark.anyio
async def test_dependent_body_expansion_without_global_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "target.py").write_text(CODE)
    llm = FakeLLM([call("search_code", query="marker"),
                   call("read_file", path="target.py", start_line=1, end_line=1),
                   call("read_file", path="target.py", start_line=1, end_line=2)],
                  generated=["marker retourne 42 [S2]."])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp,
                     planning="adaptive", max_steps=4)
    answer = await agent.ask(GOAL)
    assert answer == "marker retourne 42 [S2]."
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["local_revisions"] == 1
    assert agent.last_execution["replans"] == 0
    assert all(item["tools"] for item in llm.calls)
    assert "return 42" not in json.dumps(llm.calls[2]["messages"])
    assert "def marker" in json.dumps(llm.calls[2]["messages"])
    assert [step["evidence"] for step in agent.last_execution["plan"]] == [["S1"], ["S3"]]


@pytest.mark.anyio
async def test_actual_missing_path_error_then_discovery_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "target.py").write_text(CODE)
    goal = 'Localise marker avec search_code(query="marker", path="absent.py"), puis lis son implémentation.'
    llm = FakeLLM([call("search_code", query="marker", path="absent.py"),
                   call("list_files"), call("search_code", query="marker"),
                   call("read_file", path="target.py")],
                  generated=["marker retourne 42 [S3]."])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp,
                     planning="adaptive", max_steps=5)
    await agent.ask(goal)
    assert [item["status"] for item in agent.last_trace] == ["error", "success", "success", "success"]
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert "Path does not exist: absent.py" in json.dumps(llm.calls[1]["messages"])
    assert agent.execution.tool_calls == 4


@pytest.mark.anyio
async def test_preparatory_search_cannot_satisfy_read_or_model_completion(monkeypatch):
    agent, llm, _, client = make_agent(monkeypatch,
        [call("search_code", query="marker"), {"content": "goal_achieved"}, {"content": "goal_achieved"}],
        [search_result("target.py", "marker")], planning="adaptive", max_steps=4)
    answer = await agent.ask("Lis l'implémentation de marker.")
    assert agent.last_execution["completed"] == []
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert len(client.calls) == 1
    assert llm.generate_calls == []
    assert "Objectif final non validé" in answer


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["invented", "duplicate", "tools", "tokens", "iterations", "time"])
async def test_guards_and_shared_budgets(monkeypatch, failure):
    responses = [call("search_code", query="marker"), call("read_file", path="invented.py")]
    options = {"planning": "adaptive", "max_steps": 4}
    if failure == "duplicate":
        responses[1] = responses[0]
    if failure == "tools":
        options["max_tool_calls"] = 1
        responses[1] = call("read_file", path="target.py")
    if failure == "tokens":
        options["max_total_tokens"] = 1
    if failure == "iterations":
        options["max_steps"] = 1
    if failure == "time":
        options["max_seconds"] = 0.000001
    agent, llm, _, client = make_agent(monkeypatch, responses,
                                     [search_result("target.py", "marker")], **options)
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] != "goal_achieved"
    assert agent.execution.tool_calls <= options.get("max_tool_calls", 16)
    assert len(client.calls) <= 1
    assert llm.generate_calls == []


@pytest.mark.anyio
async def test_multisource_coverage_memory_and_current_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "target.py").write_text(CODE)
    (tmp_path / "test_target.py").write_text("from target import marker\n\ndef test_marker():\n    assert marker() == 42\n")
    memory = ConversationMemory(tmp_path / "memory", "adaptive")
    goal = "Localise marker, puis lis son implémentation, puis localise les tests associés, puis lis ces tests."
    llm = FakeLLM([call("search_code", query="def marker"), call("read_file", path="target.py"),
                   call("search_code", query="assert marker"), call("read_file", path="test_target.py")],
                  generated=["marker retourne 42 [S1].\n\nLe test vérifie marker() == 42 [S2]."])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp,
                     planning="adaptive", max_steps=5, memory=memory)
    await agent.ask(goal)
    assert agent.last_execution["completed"] == ["1", "2", "3", "4"]
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert ConversationMemory(tmp_path / "memory", "adaptive").total_turns == 1
    assert ConversationMemory(tmp_path / "memory", "other").total_turns == 0


@pytest.mark.anyio
@pytest.mark.parametrize("candidate", ["", "Je ne peux pas le déterminer à partir des sources disponibles.",
    "Voici le code [S1] :\n\n```python\ndef template():\n    pass\n```"])
async def test_final_failure_does_not_certify_collected_goal(monkeypatch, candidate):
    agent, _, _, _ = make_agent(monkeypatch, [call("read_file", path="target.py")],
                                [read_result("target.py", CODE)], generated=[candidate], planning="adaptive")
    answer = await agent.ask("Montre le code. Lis target.py.")
    assert agent.last_execution["completed"] == ["1"]
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert "def template" not in answer


def test_exact_code_needs_its_own_current_citation():
    source = build_tool_source("read_file", {"path": "target.py"}, json.dumps({"content": CODE}))
    assert answer_code_observed(f"Voici le code [S1] :\n\n```python\n{CODE}```", [source])
    assert not answer_code_observed("Voici le code [S1] :\n\n```python\ndef template():\n    pass\n```", [source])
    assert not answer_code_observed(f"Voici le code [S2] :\n\n```python\n{CODE}```", [source])


def test_requirements_do_not_predict_dependencies_or_paths():
    requirements = adaptive_requirements(GOAL)
    assert [item["success"]["tool"] for item in requirements] == ["search_code", "read_file"]
    assert all(not item["depends_on"] and "from_step" not in item["success"] for item in requirements)
    assert requirements[0]["success"]["value"] == "marker"


@pytest.mark.anyio
async def test_stale_memory_cannot_authorize_a_discovered_read(monkeypatch):
    memory = ConversationMemory()
    memory.record("marker est dans old.py", "Ancienne observation seulement.")
    agent, llm, _, client = make_agent(monkeypatch, [call("read_file", path="old.py")],
                                     planning="adaptive", memory=memory)
    await agent.ask(GOAL)
    assert "old.py" in json.dumps(llm.calls[0]["messages"])
    assert client.calls == []
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"


@pytest.mark.anyio
async def test_error_diagnosis_without_successful_read_is_not_recovery(monkeypatch):
    agent, _, _, client = make_agent(monkeypatch,
        [call("read_file", path="missing.py"), {"content": "Erreur corrigée"}, {"content": "Terminé"}],
        [tool_result(is_error=True, content={"error": "missing file"})], planning="adaptive")
    await agent.ask("Lis missing.py.")
    assert len(client.calls) == 1
    assert agent.last_execution["completed"] == []
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"


@pytest.mark.anyio
@pytest.mark.parametrize("candidate,verdict", [("marker retourne 42 [S9].", "SUPPORTED"),
    ("marker retourne 100 [S1].", "UNSUPPORTED")])
async def test_citation_and_grounding_failure_remain_failures(monkeypatch, candidate, verdict):
    agent, _, grounder, _ = make_agent(monkeypatch, [call("read_file", path="target.py")],
                                      [read_result("target.py", CODE)], generated=[candidate], planning="adaptive")
    grounder.verdict = verdict
    await agent.ask("Lis target.py.")
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert agent.last_grounding in {"citation_error", "unsupported"}


@pytest.mark.anyio
async def test_empty_search_is_preserved_without_claiming_global_absence(monkeypatch):
    agent, _, _, _ = make_agent(monkeypatch,
        [call("search_code", query="marker"), {"content": "Absent"}, {"content": "Absent"}],
        [tool_result(content={"matches": [], "truncated": False})], planning="adaptive")
    await agent.ask(GOAL)
    assert agent.last_execution["observations"] == ["S1"]
    assert agent.last_execution["completed"] == []
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"


@pytest.mark.anyio
async def test_unlisted_tool_never_reaches_mcp(monkeypatch):
    agent, _, _, client = make_agent(monkeypatch, [call("write_file", path="target.py")], planning="adaptive")
    await agent.ask("Lis target.py.")
    assert client.calls == []
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"


@pytest.mark.anyio
async def test_adaptive_preserves_conversational_recall(monkeypatch):
    memory = ConversationMemory()
    memory.record("Mon identifiant est ORION.", "Enregistré.")
    agent, _, _, client = make_agent(monkeypatch, [{"content": "ORION"}],
                                     memory=memory, planning="adaptive")
    assert await agent.ask("Quel identifiant ai-je indiqué dans cette session ?") == "ORION"
    assert agent.last_execution["mode"] == "M6_DIRECT"
    assert client.calls == []
