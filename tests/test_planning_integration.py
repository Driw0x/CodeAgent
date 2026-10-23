import asyncio
import json
from types import SimpleNamespace

import pytest

import app.agent.tool_loop as loop_module
from app.agent.tool_loop import M6_COMPLETION_RULES, ToolLoop
from app.memory.context import ContextBudget
from app.memory.conversation import ConversationMemory
from app.mcp_server import mcp
from app.tools import project_tools
from tests.test_planning import GOAL, evidence_plan
from tests.test_tool_loop import FakeClient, FakeGroundingLLM, FakeLLM, tool_result

from mcp.server import MCPServer

from app.agent.planning import validate_plan, verifies
from app.agent.tool_loop import build_tool_source
from tests.test_planning import comparison_plan, plan


def message(data):
    return {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)}


def call(name, **arguments):
    return {"role": "assistant", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def search_result(path="current.py", symbol="ORION"):
    return tool_result(content={"matches": [{"file": path, "line": 1, "text": f"{symbol} = 42"}]})


def read_result(path="current.py", content="ORION = 42"):
    return tool_result(content={"path": path, "start_line": 1, "end_line": 1, "content": content})


class PlanningClient(FakeClient):
    async def list_tools(self):
        tools = (await super().list_tools()).tools
        tools.append(SimpleNamespace(name="search_code", description="Search code.", input_schema={
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
        }))
        return SimpleNamespace(tools=tools)


def make_agent(monkeypatch, responses, results=(), generated=None, **options):
    client = PlanningClient(results)
    monkeypatch.setattr(loop_module, "Client", lambda server: client)
    llm = FakeLLM(responses, generated=generated)
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object(), **options)
    return agent, llm, grounding, client


@pytest.mark.anyio
async def test_direct_request_uses_no_planner_and_preserves_memory_recall(monkeypatch):
    memory = ConversationMemory()
    memory.record("Mon identifiant de session est ORION.", "Enregistré.")
    agent, llm, grounding, client = make_agent(monkeypatch, [{"role": "assistant", "content": "ORION"}], memory=memory)
    assert await agent.ask("Quel identifiant ai-je indiqué dans cette session ?") == "ORION"
    assert agent.last_execution["mode"] == "M6_DIRECT"
    assert len(llm.calls) == 1
    assert "ORION" in json.dumps(llm.calls[0]["messages"])
    assert client.calls == grounding.calls == []
    assert agent.last_stage_stats["planner"] == {"prompt_tokens": 0, "completion_tokens": 0}


@pytest.mark.anyio
async def test_planned_mcp_execution_has_validated_dependencies_grounding_and_single_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "current.py").write_text("ORION = 42\n", encoding="utf-8")
    memory = ConversationMemory(tmp_path / "memory", "planned")
    llm = FakeLLM([message(evidence_plan()), call("search_code", query="ORION"),
                   call("read_file", path="current.py")], generated=["ORION vaut 42 [S1]."])
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=mcp, memory=memory)
    assert await agent.ask(GOAL) == "ORION vaut 42 [S1]."
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["remaining"] == []
    assert agent.last_execution["current_step"] is None
    final_prompt = llm.generate_calls[0]["prompt"]
    assert "ÉTAT MESURÉ DE LA COLLECTE" in final_prompt
    assert final_prompt.count('"status": "COMPLETED"') == 2
    assert all(step["objective"] in final_prompt for step in agent.last_execution["plan"])
    assert [trace["step_id"] for trace in agent.last_trace] == ["1", "2"]
    assert [trace["status"] for trace in agent.last_trace] == ["success", "success"]
    events = agent.last_execution["events"]
    initial = next(event for event in events if event["event"] == "plan")
    assert [step["status"] for step in initial["plan"]] == ["PENDING", "PENDING"]
    assert [event["step_id"] for event in events if event["event"] == "step_completed"] == ["1", "2"]
    assert next(i for i, event in enumerate(events) if event["event"] == "step_completed") < next(
        i for i, event in enumerate(events) if event["event"] == "step_started" and event["step_id"] == "2")
    assert llm.calls[0]["tools"] is None
    assert [tool["function"]["name"] for tool in llm.calls[1]["tools"]] == ["search_code"]
    assert [tool["function"]["name"] for tool in llm.calls[2]["tools"]] == ["read_file"]
    decision_context = llm.calls[2]["messages"]
    assert decision_context[-2]["role"] == "tool"
    assert decision_context[-1]["role"] == "user"
    assert json.dumps(agent.last_execution["plan"][1]["objective"], ensure_ascii=False) in decision_context[-1]["content"]
    assert '"id": "2"' in decision_context[-1]["content"]
    assert agent.last_debug["source_mapping"] == {"S1": "S2"}
    assert "ORION = 42" in grounding.calls[0]["prompt"]
    entries = (tmp_path / "memory/history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(entries) == 1
    assert len(json.loads(entries[0])["observations"]) == 4
    assert ConversationMemory(tmp_path / "memory", "planned").total_turns == 1
    assert "M8_PLANNED" not in ConversationMemory(tmp_path / "memory", "other").recall("ORION")


@pytest.mark.anyio
@pytest.mark.parametrize("content", ["not JSON", json.dumps({"goal": GOAL, "steps": []})])
async def test_invalid_planner_output_stops_without_tool_calls(monkeypatch, content):
    agent, llm, _, client = make_agent(monkeypatch, [{"role": "assistant", "content": content}] * 2)
    result = await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "invalid_plan"
    assert "Objectif final non validé" in result
    assert client.calls == llm.generate_calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("response", [
    call("read_file", path="invented.py"),
    {"role": "assistant", "tool_calls": [call("search_code", query="ORION")["tool_calls"][0],
                                             call("read_file", path="invented.py")["tool_calls"][0]]},
])
async def test_unvalidated_or_batched_tools_never_execute(monkeypatch, response):
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), response, response])
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert client.calls == []
    assert agent.last_execution["completed"] == []


@pytest.mark.anyio
async def test_dependent_read_cannot_invent_discovered_path(monkeypatch):
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
                                                 call("read_file", path="invented.py")], [search_result()])
    await agent.ask(GOAL)
    assert len(client.calls) == 1
    assert agent.last_execution["completed"] == ["1"]
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"


@pytest.mark.anyio
@pytest.mark.parametrize("from_step", [False, True])
async def test_read_cannot_substitute_another_steps_explicit_path(monkeypatch, from_step):
    goal = "Lis a.py, puis lis b.py."
    condition = {"tool": "read_file", "kind": "nonempty"}
    plan = {"goal": goal, "steps": [
        {"id": "1", "objective": "Lis a.py", "depends_on": [], "success": dict(condition)},
        {"id": "2", "objective": "lis b.py", "depends_on": ["1"],
         "success": {**condition, **({"from_step": "1"} if from_step else {})}},
    ]}
    agent, llm, grounding, client = make_agent(monkeypatch,
        [message(plan), call("read_file", path="a.py", start_line=1, end_line=1),
         call("read_file", path="a.py", start_line=2, end_line=2)],
        [tool_result(content={"path": "a.py", "start_line": 1, "end_line": 1, "content": "A = 1"}),
         tool_result(content={"path": "a.py", "start_line": 2, "end_line": 2, "content": "C = 3"})],
        generated=["A vaut 1 [S1]. C vaut 3 [S2]."])
    answer = await agent.ask(goal)
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert agent.last_execution["completed"] == ["1"]
    assert agent.last_execution["remaining"] == ["2"]
    assert len(client.calls) == 1
    assert llm.generate_calls == grounding.calls == []
    assert "Objectif final non validé" in answer


@pytest.mark.anyio
async def test_explicit_read_steps_collect_each_requested_file(monkeypatch):
    goal = "Lis a.py, puis lis b.py."
    plan = {"goal": goal, "steps": [
        {"id": "1", "objective": "Lis a.py", "depends_on": [],
         "success": {"tool": "read_file", "kind": "nonempty"}},
        {"id": "2", "objective": "lis b.py", "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty"}},
    ]}
    agent, _, _, client = make_agent(monkeypatch,
        [message(plan), call("read_file", path="a.py"), call("read_file", path="b.py")],
        [read_result("a.py", "A = 1"), read_result("b.py", "B = 2")],
        generated=["A vaut 1 [S1]. B vaut 2 [S2]."])
    assert await agent.ask(goal) == "A vaut 1 [S1]. B vaut 2 [S2]."
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert [arguments["path"] for _, arguments in client.calls] == ["a.py", "b.py"]
    assert agent.last_execution["completed"] == ["1", "2"]


@pytest.mark.anyio
async def test_explicit_read_target_and_observed_origin_can_agree(monkeypatch):
    goal = "Localise ORION dans current.py, puis lis current.py."
    plan = evidence_plan(goal)
    plan["steps"][0]["objective"] = "Localise ORION dans current.py"
    plan["steps"][1]["objective"] = "lis current.py"
    agent, _, _, client = make_agent(monkeypatch,
        [message(plan), call("search_code", query="ORION"), call("read_file", path="current.py")],
        [search_result(), read_result()], generated=["ORION vaut 42 [S1]."])
    assert await agent.ask(goal) == "ORION vaut 42 [S1]."
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert len(client.calls) == 2


@pytest.mark.anyio
async def test_ambiguous_search_requires_user_decision(monkeypatch):
    result = tool_result(content={"matches": [
        {"file": path, "line": 1, "text": "ORION = 42"} for path in ["one.py", "two.py"]
    ]})
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
                                                 call("read_file", path="one.py")], [result])
    answer = await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "user_decision"
    assert "ambigu" in answer
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_recoverable_error_retries_without_replanning(monkeypatch):
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py", start_line=999),
        call("read_file", path="current.py", start_line=1, end_line=1)], [search_result(),
        tool_result(is_error=True, content={"message": "start_line exceeds file length (1 lines)."}), read_result()],
        generated=["ORION vaut 42 [S1]."])
    await agent.ask(GOAL)
    assert [trace["status"] for trace in agent.last_trace] == ["success", "error", "success"]
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["replans"] == 0
    assert len(client.calls) == 3


@pytest.mark.anyio
async def test_replanning_preserves_validated_step_and_uses_new_observations(monkeypatch):
    replanned = evidence_plan()
    replanned["steps"][1]["success"] = {"tool": "read_file", "kind": "contains", "value": "ORION", "from_step": "1"}
    agent, llm, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py", start_line=1, end_line=1), message(replanned),
        call("read_file", path="current.py", start_line=2, end_line=2)], [tool_result(content={
        "matches": [{"file": "current.py", "line": 2, "text": "ORION = 42"}]}),
        read_result(content="OTHER = 1"), tool_result(content={"path": "current.py", "start_line": 2,
        "end_line": 2, "content": "ORION = 42"})], generated=["ORION vaut 42 [S2]."])
    await agent.ask(GOAL)
    assert agent.last_execution["replans"] == 1
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["plan"][0]["evidence"] == ["S1"]
    assert sum(name == "search_code" for name, _ in client.calls) == 1
    replan = next(event for event in agent.last_execution["events"] if event["event"] == "replanning")
    assert replan["preserved"] == ["1"]
    assert any(event["event"] == "step_failed" for event in agent.last_execution["events"])
    assert "OTHER = 1" in json.dumps(llm.calls[3]["messages"])


@pytest.mark.anyio
async def test_replanning_cannot_rewrite_completed_steps(monkeypatch):
    invalid = evidence_plan()
    invalid["steps"][0]["success"]["kind"] = "nonempty"
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py"), message(invalid), message(invalid)], [search_result(), read_result(content="OTHER = 1")])
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "invalid_plan"
    assert agent.last_execution["completed"] == ["1"]
    assert len(client.calls) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("options,reason", [
    ({"max_replans": 0}, "replan_limit"),
    ({"max_steps": 1}, "step_limit"),
    ({"max_tool_calls": 1}, "global_budget"),
])
async def test_step_replan_and_tool_budgets_stop_without_false_success(monkeypatch, options, reason):
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py")], [search_result(), read_result(content="OTHER = 1")], **options)
    result = await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == reason
    assert "Objectif final non validé" in result
    assert agent.last_execution["completed"] == ["1"]
    assert agent.last_execution["remaining"] == ["2"]
    assert len(client.calls) <= options.get("max_tool_calls", 2)


@pytest.mark.anyio
async def test_duplicate_call_is_not_reexecuted_across_planning_steps(monkeypatch):
    plan = evidence_plan()
    plan["steps"][1]["success"] = {"tool": "search_code", "kind": "nonempty"}
    agent, _, _, client = make_agent(monkeypatch, [message(plan), call("search_code", query="ORION"),
                                                 call("search_code", query="ORION")], [search_result()])
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "repeated_errors"
    assert len(client.calls) == 1
    assert [trace["status"] for trace in agent.last_trace] == ["success", "duplicate"]


@pytest.mark.anyio
async def test_planner_token_budget_is_checked_before_model_call(monkeypatch):
    agent, llm, _, client = make_agent(monkeypatch, [], max_total_tokens=1)
    result = await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "global_budget"
    assert llm.calls == client.calls == []
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_planner_context_budget_is_checked_before_model_call(monkeypatch):
    agent, llm, _, client = make_agent(monkeypatch, [])
    agent.budget = ContextBudget(1000, 200)
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "context_budget"
    assert llm.calls == client.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("at_planner", [True, False])
async def test_user_decision_is_explicit_and_preserves_unfinished_work(monkeypatch, at_planner):
    decision = message({"stop": "user_decision", "reason": "Quel projet faut-il analyser ?"})
    responses = [decision] if at_planner else [message(evidence_plan()), decision]
    agent, _, _, client = make_agent(monkeypatch, responses)
    result = await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "user_decision"
    assert "Quel projet" in result
    assert client.calls == []
    assert not any(event["event"] == "tool_reminder" for event in agent.last_execution["events"])


@pytest.mark.anyio
@pytest.mark.parametrize("candidate,verdict,grounding", [
    ("ORION améliore les performances [S1].", "UNSUPPORTED", "unsupported"),
    ("ORION vaut 42 [S99].", "SUPPORTED", "citation_error"),
    ("Je ne peux pas le déterminer à partir des sources disponibles.", "SUPPORTED", "not_used"),
])
async def test_final_rejection_or_abstention_never_marks_goal_achieved(monkeypatch, candidate, verdict, grounding):
    agent, _, verifier, _ = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py")], [search_result(), read_result()], generated=[candidate])
    verifier.verdict = verdict
    result = await agent.ask(GOAL)
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert agent.last_grounding == grounding
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_current_sources_override_historical_project_facts(monkeypatch):
    memory = ConversationMemory()
    memory.record("Lis ORION dans current.py", "ORION vaut 41 [S1].")
    agent, _, grounding, _ = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py")], [search_result(), read_result()], memory=memory,
        generated=["ORION vaut 42 [S1]."])
    assert await agent.ask(GOAL) == "ORION vaut 42 [S1]."
    assert "ORION = 42" in grounding.calls[0]["prompt"]
    assert "ORION vaut 41" not in grounding.calls[0]["prompt"]
    assert memory.total_turns == 2


@pytest.mark.anyio
async def test_real_mcp_schema_failure_is_recoverable_without_replanning(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "current.py").write_text("ORION = 42\n", encoding="utf-8")
    llm = FakeLLM([message(evidence_plan()), call("search_code", query=7),
        call("search_code", query="ORION"), call("read_file", path="current.py")],
        generated=["ORION vaut 42 [S2]."])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp)
    assert await agent.ask(GOAL) == "ORION vaut 42 [S2]."
    assert [trace["status"] for trace in agent.last_trace] == ["error", "success", "success"]
    assert agent.last_trace[0]["arguments"]["query"] == 7
    assert agent.last_execution["replans"] == 0
    assert agent.last_execution["stop_reason"] == "goal_achieved"


@pytest.mark.anyio
@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("controlled MCP failure")])
async def test_timeout_and_exception_use_m6_recovery_before_replanning(monkeypatch, error):
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py", start_line=1, end_line=1),
        call("read_file", path="current.py", start_line=1, end_line=2)], [search_result(), read_result()],
        generated=["ORION vaut 42 [S1]."])
    original_call = client.call_tool
    failed = False

    async def call_once_fails(name, arguments):
        nonlocal failed
        if name == "read_file" and not failed:
            failed = True
            client.calls.append((name, arguments))
            raise error
        return await original_call(name, arguments)

    client.call_tool = call_once_fails
    await agent.ask(GOAL)
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["replans"] == 0
    assert agent.last_trace[1]["status"] == ("timeout" if isinstance(error, TimeoutError) else "exception")
    assert len(client.calls) == 3


@pytest.mark.anyio
async def test_exhausted_retries_trigger_one_replan_and_preserve_completed_step(monkeypatch):
    replacement = evidence_plan()
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py", start_line=20),
        call("read_file", path="current.py", start_line=30), message(replacement),
        call("read_file", path="current.py", start_line=1, end_line=1)], [search_result(),
        tool_result(is_error=True, content={"message": "start_line exceeds file length (1 lines)."}),
        tool_result(is_error=True, content={"message": "start_line exceeds file length (1 lines)."}), read_result()],
        generated=["ORION vaut 42 [S1]."], max_steps=5)
    result = await agent.ask(GOAL)
    assert agent.last_execution["replans"] == 1
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["plan"][0]["evidence"] == ["S1"]
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert [trace["status"] for trace in agent.last_trace] == ["success", "error", "error", "success"]
    assert len(client.calls) == 4
    assert result == "ORION vaut 42 [S1]."


@pytest.mark.anyio
async def test_permanent_error_after_replanning_stops_at_replan_limit(monkeypatch):
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py", start_line=20), message(evidence_plan()),
        call("read_file", path="current.py", start_line=30)], [search_result(),
        tool_result(is_error=True, content={"message": "controlled permanent failure"}),
        tool_result(is_error=True, content={"message": "controlled permanent failure"})],
        max_retries=0, max_replans=1)
    result = await agent.ask(GOAL)
    assert agent.last_execution["replans"] == 1
    assert agent.last_execution["completed"] == ["1"]
    assert agent.last_execution["remaining"] == ["2"]
    assert agent.last_execution["stop_reason"] == "replan_limit"
    assert agent.last_execution["current_step"] is None
    assert agent.last_execution["plan"][1]["status"] == "FAILED"
    assert len(client.calls) == 3
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_replanned_step_receives_one_fresh_retry_without_repeating_old_calls(monkeypatch):
    failure = tool_result(is_error=True, content={"message": "controlled read failure"})
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py", start_line=20),
        call("read_file", path="current.py", start_line=30), message(evidence_plan()),
        call("read_file", path="current.py", start_line=40),
        call("read_file", path="current.py", start_line=1, end_line=1)],
        [search_result(), failure, failure, failure, read_result()],
        generated=["ORION vaut 42 [S1]."], max_retries=1, max_replans=1, max_steps=6)
    assert await agent.ask(GOAL) == "ORION vaut 42 [S1]."
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["replans"] == 1
    assert [trace["status"] for trace in agent.last_trace] == ["success", "error", "error", "error", "success"]
    assert len(client.calls) == 5
    assert len({json.dumps(call, sort_keys=True) for call in client.calls}) == 5


@pytest.mark.anyio
async def test_explicit_planning_off_preserves_direct_m6_for_complex_request(monkeypatch):
    agent, llm, _, client = make_agent(monkeypatch, [call("read_file", path="current.py"),
        {"role": "assistant", "content": "Terminé"}], [read_result()],
        generated=["ORION vaut 42 [S1]."], planning="off")
    assert await agent.ask(GOAL) == "ORION vaut 42 [S1]."
    assert agent.last_execution["mode"] == "M6_DIRECT"
    assert agent.last_execution["plan"] == []
    assert agent.last_stage_stats["planner"]["prompt_tokens"] == 0
    assert all(response["tools"] is not None for response in llm.calls)
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_planner_preserves_latest_user_decision_without_cross_session_data(tmp_path, monkeypatch):
    memory = ConversationMemory(tmp_path / "memory", "selected")
    memory.record("Décision : analyser ORION, ne pas modifier les fichiers.", "Enregistré.")
    memory.record("Correction : analyser VEGA, conserver le mode read-only.", "Enregistré.")
    other = ConversationMemory(tmp_path / "memory", "other")
    other.record("Secret d'une autre session : CASSIOPEIA.", "Enregistré.")
    agent, llm, _, _ = make_agent(monkeypatch, [message({"stop": "user_decision", "reason": "Choisir la cible"})], memory=memory)
    await agent.ask(GOAL)
    context = json.dumps(llm.calls[0]["messages"], ensure_ascii=False)
    assert "VEGA" in context
    assert "read-only" in context
    assert "CASSIOPEIA" not in context
    assert ConversationMemory(tmp_path / "memory", "selected").total_turns == 3
    assert ConversationMemory(tmp_path / "memory", "other").total_turns == 1


@pytest.mark.anyio
@pytest.mark.parametrize("question", [GOAL, "Réponds uniquement par OK."])
async def test_real_mcp_budget_stop_survives_taskgroup_wrapping(question):
    llm = FakeLLM([])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp, max_total_tokens=1)
    result = await agent.ask(question)
    assert agent.last_execution["stop_reason"] == "global_budget"
    assert "Objectif final non validé" in result
    assert llm.calls == []
    assert agent.memory.total_turns == 1


@pytest.mark.anyio
async def test_actual_wait_for_timeout_preserves_failed_step_and_stops_cleanly(monkeypatch):
    agent, _, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION")],
        [search_result()], tool_timeout=0.001, max_retries=0, max_replans=0)
    client.delay = 0.05
    result = await agent.ask(GOAL)
    assert agent.last_trace[0]["status"] == "timeout"
    assert agent.last_trace[0]["duration_seconds"] > 0
    assert agent.last_execution["stop_reason"] == "replan_limit"
    assert agent.last_execution["completed"] == []
    assert agent.last_execution["plan"][0]["status"] == "FAILED"
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_omitted_explicit_objective_cannot_be_announced_as_goal_achieved(monkeypatch):
    goal = "Localise ORION, puis lis son implémentation et vérifie les tests associés."
    incomplete = evidence_plan()
    incomplete["goal"] = goal
    agent, _, _, _ = make_agent(monkeypatch, [message(incomplete), call("search_code", query="ORION"),
        call("read_file", path="current.py")], [search_result(), read_result()],
        generated=["ORION vaut 42 [S1]."])
    result = await agent.ask(goal)
    assert agent.last_execution["stop_reason"] != "goal_achieved"
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_final_context_cannot_drop_step_evidence_and_still_claim_success(monkeypatch):
    def reduced_sources(question, sources, *args, **kwargs):
        source = dict(sources[0])
        source["content"] = 'Result:\n' + json.dumps({
            "path": "current.py", "start_line": 1, "end_line": 2, "truncated": True,
            "content_excerpts": [{"line": 2, "text": "OTHER = 1"}],
        })
        return [source], "Only OTHER = 1 remains in the final context."

    monkeypatch.setattr(loop_module, "fit_sources", reduced_sources)
    agent, _, _, _ = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py")], [search_result(), tool_result(content={
            "path": "current.py", "start_line": 1, "end_line": 2, "content": "ORION = 42\nOTHER = 1",
        })], generated=["OTHER vaut 1 [S1]."])
    result = await agent.ask(GOAL)
    assert agent.last_grounding == "passed"
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_read_without_origin_can_only_use_path_explicitly_requested(monkeypatch):
    goal = "Localise ORION dans current.py, puis lis current.py."
    plan = evidence_plan(goal)
    plan["steps"][0]["objective"] = "Localise ORION dans current.py"
    plan["steps"][1]["objective"] = "lis current.py"
    plan["steps"][1]["success"].pop("from_step")
    agent, _, _, client = make_agent(monkeypatch, [message(plan), call("search_code", query="ORION"),
        call("read_file", path="invented.py")], [search_result()])
    await agent.ask(goal)
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_global_time_budget_cancels_mcp_initialization_before_llm_or_tools(monkeypatch):
    class SlowClient(PlanningClient):
        async def __aenter__(self):
            await asyncio.sleep(0.05)
            return self

    client = SlowClient()
    monkeypatch.setattr(loop_module, "Client", lambda server: client)
    llm = FakeLLM([])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=object(), max_seconds=0.001)
    result = await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "global_budget"
    assert agent.last_execution["budgets"]["seconds"] == 0
    assert llm.calls == client.calls == []
    assert agent.last_trace == []
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_recorded_plan_accepts_case_but_never_executes_an_unobserved_path(monkeypatch):
    # Exact Devstral response captured in planner-only.json on 2026-10-15.
    content = '''```json
{
  "goal": "Localise marker, puis lis son implémentation.",
  "steps": [
    {
      "id": "1",
      "objective": "Localise marker",
      "depends_on": [],
      "success": {
        "tool": "search_code",
        "kind": "contains",
        "value": "marker"
      }
    },
    {
      "id": "2",
      "objective": "Lis son implémentation",
      "depends_on": ["1"],
      "success": {
        "tool": "read_file",
        "kind": "nonempty",
        "from_step": "1",
        "path": "chemin_vers_marker"
      }
    }
  ]
}
```'''
    response = {"role": "assistant", "content": content}
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    agent, llm, grounding, client = make_agent(monkeypatch, [response,
        call("search_code", query="marker"), call("read_file", path="chemin_vers_marker")],
        [search_result(symbol="marker")])
    result = await agent.ask("Localise marker, puis lis son implémentation.")
    assert agent.last_execution["stop_reason"] == "user_decision"
    assert "Chemin observé absent ou ambigu" in result
    assert agent.last_execution["plan"][1]["objective"] == "lis son implémentation"
    assert agent.last_execution["completed"] == ["1"]
    assert len(client.calls) == 1 and client.calls[0][0] == "search_code"
    assert grounding.calls == llm.generate_calls == []
    assert len(agent.last_trace) == 1
    assert len(llm.calls) == 3
    assert agent.last_debug["planner_responses"] == [response]
    assert agent.last_debug["planner_responses"][0]["content"] == content


@pytest.mark.anyio
async def test_unique_observed_definition_disambiguates_recorded_real_search(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "target.py").write_text("def marker():\n    return 42\n", encoding="utf-8")
    (tmp_path / "test_target.py").write_text(
        "from target import marker\n\ndef test_marker():\n    assert marker() == 42\n", encoding="utf-8")
    goal = "Localise marker, puis lis son implémentation."
    plan = {"goal": goal, "steps": [
        {"id": "1", "objective": "Localise marker", "depends_on": [],
         "success": {"tool": "search_code", "kind": "contains", "value": "marker"}},
        {"id": "2", "objective": "lis son implémentation", "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "1"}},
    ]}
    candidate = "marker est défini dans target.py [S1].\n\nmarker retourne 42 [S2]."
    llm = FakeLLM([message(plan), call("search_code", query="marker"),
                   call("read_file", path="target.py")], generated=[candidate])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp)
    assert await agent.ask(goal) == candidate
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["replans"] == 0
    assert [trace["tool"] for trace in agent.last_trace] == ["search_code", "read_file"]
    assert agent.last_trace[1]["arguments"]["path"] == "target.py"
    search = agent.last_debug["sources_before_selection"][0]
    assert search["search_paths"] == ["target.py", "test_target.py"]
    assert search["definition_paths"] == ["target.py"]
    observed = json.loads(agent.memory.recent[-1]["observations"][1]["content"])
    assert json.loads(observed["result"])["matches"] == [
        {"file": "target.py", "line": 1, "text": "def marker():"},
        {"file": "test_target.py", "line": 1, "text": "from target import marker"},
        {"file": "test_target.py", "line": 3, "text": "def test_marker():"},
        {"file": "test_target.py", "line": 4, "text": "assert marker() == 42"},
    ]


@pytest.mark.anyio
async def test_declaration_only_read_replans_then_validates_body_without_repeating_search(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "target.py").write_text("def marker():\n    return 42\n", encoding="utf-8")
    goal = "Localise marker, puis lis son implémentation."
    plan = {"goal": goal, "steps": [
        {"id": "1", "objective": "Localise marker", "depends_on": [],
         "success": {"tool": "search_code", "kind": "contains", "value": "marker"}},
        {"id": "2", "objective": "lis son implémentation", "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "1"}},
    ]}
    llm = FakeLLM([message(plan), call("search_code", query="marker"),
        call("read_file", path="target.py", start_line=1, end_line=1), message(plan),
        call("read_file", path="target.py", start_line=1, end_line=2)],
        generated=["marker retourne 42 [S2]."])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp)
    assert await agent.ask(goal) == "marker retourne 42 [S2]."
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["replans"] == 1
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["plan"][0]["evidence"] == ["S1"]
    assert agent.last_execution["plan"][1]["evidence"] == ["S3"]
    assert [trace["status"] for trace in agent.last_trace] == ["success", "success", "success"]
    assert [trace["tool"] for trace in agent.last_trace].count("search_code") == 1
    assert agent.last_trace[1]["arguments"]["end_line"] == 1
    assert agent.last_trace[2]["arguments"]["end_line"] == 2
    events = agent.last_execution["events"]
    failed = next(index for index, event in enumerate(events) if event["event"] == "step_failed")
    replanned = next(index for index, event in enumerate(events) if event["event"] == "replanning")
    completed = next(index for index, event in enumerate(events)
                     if event["event"] == "step_completed" and event["step_id"] == "2")
    assert failed < replanned < completed
    assert events[replanned]["preserved"] == ["1"]


@pytest.mark.anyio
@pytest.mark.parametrize("resume_with_tool", [True, False])
async def test_executor_receives_replan_diagnostic_and_requires_actual_corrected_observation(
    tmp_path, monkeypatch, resume_with_tool,
):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "current.py").write_text("def ORION():\n    return 42\n", encoding="utf-8")
    resume = call("read_file", path="current.py", start_line=1, end_line=2) if resume_with_tool else {
        "role": "assistant", "content": "Je vais maintenant lire les lignes suivantes pour obtenir le corps de cette fonction.",
    }
    llm = FakeLLM([message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py", start_line=1, end_line=1), message(evidence_plan()), resume,
        *([] if resume_with_tool else [resume])],
        generated=["ORION retourne 42 [S2]."])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp)
    answer = await agent.ask(GOAL)
    diagnostic = agent.last_execution["errors"][0]
    assert "inclure le corps" in diagnostic
    initial_system = llm.calls[2]["messages"][0]["content"]
    resumed_system = llm.calls[4]["messages"][0]["content"]
    assert diagnostic not in initial_system
    assert diagnostic in resumed_system
    assert agent.budget.measure(llm.calls[4]["messages"], llm.calls[4]["tools"]) <= agent.budget.input_limit
    assert agent.last_execution["replans"] == 1
    assert agent.last_execution["plan"][0]["evidence"] == ["S1"]
    assert [trace["tool"] for trace in agent.last_trace].count("search_code") == 1
    if resume_with_tool:
        assert agent.last_execution["stop_reason"] == "goal_achieved"
        assert agent.last_execution["plan"][1]["evidence"] == ["S3"]
        assert answer == "ORION retourne 42 [S2]."
    else:
        assert agent.last_execution["stop_reason"] == "insufficient_evidence"
        assert agent.last_execution["completed"] == ["1"]
        assert agent.last_execution["plan"][1]["status"] == "FAILED"
        assert len(agent.last_trace) == 2
        assert llm.generate_calls == []
        assert "Objectif final non validé" in answer


@pytest.mark.anyio
async def test_m6_completion_rules_do_not_stop_required_m8_collection_early(monkeypatch):
    direct, direct_llm, _, _ = make_agent(monkeypatch, [{"role": "assistant", "content": "OK"}])
    assert await direct.ask("Réponds uniquement par OK.") == "OK"
    direct_system = direct_llm.calls[0]["messages"][0]["content"]
    for rule in M6_COMPLETION_RULES.splitlines():
        assert rule in direct_system

    goal = "Localise ORION, puis lis son implémentation, puis vérifie les tests associés."
    plan = evidence_plan(goal)
    plan["steps"].append({"id": "3", "objective": "vérifie les tests associés", "depends_on": ["2"],
                          "success": {"tool": "search_code", "kind": "nonempty"}})
    prose = {"role": "assistant", "content": "ORION vaut 42. Je vais maintenant vérifier les tests associés."}
    agent, llm, grounding, client = make_agent(monkeypatch, [message(plan), call("search_code", query="ORION"),
        call("read_file", path="current.py"), prose, prose],
        [search_result(), read_result()])
    answer = await agent.ask(goal)
    for request in llm.calls[1:]:
        system = request["messages"][0]["content"]
        for rule in M6_COMPLETION_RULES.splitlines():
            assert rule not in system
        assert "L'étape courante exige un appel réel de son tool" in system
        assert "après validation du plan" in system
        assert "la boucle contrôle" in system
    assert len(llm.calls) == 5
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["remaining"] == ["3"]
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert len(client.calls) == 2
    assert llm.generate_calls == grounding.calls == []
    assert "Objectif final non validé" in answer


@pytest.mark.anyio
@pytest.mark.parametrize("recovers", [False, True])
async def test_grouped_calls_get_one_reminder_without_executing_or_replaying_them(monkeypatch, recovers):
    grouped = {"role": "assistant", "tool_calls": [
        call("search_code", query="ORION")["tool_calls"][0],
        call("read_file", path="unobserved.py")["tool_calls"][0],
    ]}
    retry = [call("search_code", query="ORION"), call("read_file", path="current.py")] if recovers else [grouped]
    agent, llm, grounding, client = make_agent(monkeypatch, [message(evidence_plan()), grouped, *retry],
        [search_result(), read_result()], generated=["ORION vaut 42 [S1]."])
    answer = await agent.ask(GOAL)
    reminders = [event for event in agent.last_execution["events"] if event["event"] == "tool_reminder"]
    assert len(reminders) == 1 and reminders[0]["reason"] == "grouped_tool_calls"
    assert not any(message.get("tool_calls") for message in llm.calls[2]["messages"])
    assert llm.calls[2]["messages"][-1]["role"] == "user"
    assert agent.last_execution["replans"] == 0
    assert agent.last_steps == (3 if recovers else 2)
    assert len(client.calls) == (2 if recovers else 0)
    assert all(arguments.get("path") != "unobserved.py" for _, arguments in client.calls)
    if recovers:
        assert answer == "ORION vaut 42 [S1]."
        assert agent.last_execution["stop_reason"] == "goal_achieved"
        assert [trace["step_id"] for trace in agent.last_trace] == ["1", "2"]
    else:
        assert agent.last_execution["stop_reason"] == "insufficient_evidence"
        assert agent.last_execution["completed"] == []
        assert "Objectif final non validé" in answer
        assert llm.generate_calls == grounding.calls == []


@pytest.mark.anyio
async def test_text_after_completed_observations_gets_one_reminder_for_current_step(monkeypatch):
    goal = "Localise ORION, puis lis son implémentation, puis vérifie les tests associés."
    plan = evidence_plan(goal)
    plan["steps"].append({"id": "3", "objective": "vérifie les tests associés", "depends_on": ["2"],
                          "success": {"tool": "search_code", "kind": "contains", "value": "ORION"}})
    prose = {"role": "assistant", "content": "ORION vaut 42 [S2]. Je vais vérifier les tests associés."}
    candidate = "ORION vaut 42 [S1].\n\nUn test compare ORION à 42 [S2]."
    agent, llm, _, client = make_agent(monkeypatch, [message(plan), call("search_code", query="ORION"),
        call("read_file", path="current.py"), prose, call("search_code", query="assert ORION")],
        [search_result(), read_result(), tool_result(content={"matches": [
            {"file": "test_current.py", "line": 1, "text": "assert ORION == 42"}]})],
        generated=[candidate], max_steps=6)
    assert await agent.ask(goal) == candidate
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["completed"] == ["1", "2", "3"]
    assert len(client.calls) == 3
    assert agent.last_steps == 4
    reminder = next(event for event in agent.last_execution["events"] if event["event"] == "tool_reminder")
    assert reminder["step_id"] == "3"
    after_two_tools = llm.calls[3]["messages"]
    assert after_two_tools[-2]["role"] == "tool"
    assert after_two_tools[-1]["role"] == "user"
    assert '"id": "3"' in after_two_tools[-1]["content"]
    resumed = llm.calls[4]
    assert resumed["messages"][-3] == prose
    assert resumed["messages"][-1]["role"] == "user"
    assert [tool["function"]["name"] for tool in resumed["tools"]] == ["search_code"]
    observations = agent.memory.recent[-1]["observations"]
    assert len(observations) == 6
    assert all(message.get("tool_calls") for message in observations if message["role"] == "assistant")


@pytest.mark.anyio
async def test_text_after_replan_gets_one_reminder_without_validating_unread_body(monkeypatch):
    prose = {"role": "assistant", "content": "Le corps n'a pas été lu. Je vais élargir la lecture pour inclure le corps de la fonction."}
    body = tool_result(content={"path": "current.py", "start_line": 1, "end_line": 2,
                               "content": "def ORION():\n    return 42",
                               "complete_symbols": [{"name": "ORION", "start_line": 1, "end_line": 2}]})
    agent, llm, _, client = make_agent(monkeypatch, [message(evidence_plan()), call("search_code", query="ORION"),
        call("read_file", path="current.py", start_line=1, end_line=1), message(evidence_plan()), prose,
        call("read_file", path="current.py", start_line=1, end_line=2)],
        [search_result(), read_result(content="def ORION():"), body],
        generated=["ORION retourne 42 [S2]."], max_steps=6)
    assert await agent.ask(GOAL) == "ORION retourne 42 [S2]."
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["replans"] == 1
    assert agent.last_execution["plan"][0]["evidence"] == ["S1"]
    assert agent.last_execution["plan"][1]["evidence"] == ["S3"]
    assert [trace["step_id"] for trace in agent.last_trace] == ["1", "2", "2"]
    assert len(client.calls) == 3
    assert agent.last_steps == 4
    assert agent.last_execution["budgets"]["iterations"] == 2
    assert llm.calls[5]["messages"][-3] == prose
    assert agent.budget.measure(llm.calls[5]["messages"], llm.calls[5]["tools"]) <= agent.budget.input_limit
    assert len(agent.memory.recent[-1]["observations"]) == 6


@pytest.mark.anyio
async def test_reminder_is_not_reset_when_same_step_is_replanned(monkeypatch):
    prose = {"role": "assistant", "content": "Je vais lire le corps de ORION."}
    agent, llm, grounding, client = make_agent(monkeypatch, [message(evidence_plan()),
        call("search_code", query="ORION"), prose,
        call("read_file", path="current.py", start_line=1, end_line=1), message(evidence_plan()), prose],
        [search_result(), read_result(content="def ORION():")], max_steps=6)
    answer = await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert agent.last_execution["completed"] == ["1"]
    assert agent.last_execution["plan"][1]["evidence"] == []
    assert agent.last_execution["replans"] == 1
    assert [event["step_id"] for event in agent.last_execution["events"]
            if event["event"] == "tool_reminder"] == ["2"]
    assert "Après un rappel" in answer
    assert len(client.calls) == 2
    assert agent.last_steps == 4
    assert llm.generate_calls == grounding.calls == []
    assert len(agent.memory.recent[-1]["observations"]) == 4


@pytest.mark.anyio
async def test_reminder_does_not_allow_duplicate_mcp_call(monkeypatch):
    plan = evidence_plan()
    plan["steps"][1]["success"] = {"tool": "search_code", "kind": "nonempty"}
    agent, llm, grounding, client = make_agent(monkeypatch, [message(plan),
        call("search_code", query="ORION"), {"role": "assistant", "content": "Je vais chercher ORION."},
        call("search_code", query="ORION")], [search_result()])
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "repeated_errors"
    assert agent.last_execution["completed"] == ["1"]
    assert [trace["status"] for trace in agent.last_trace] == ["success", "duplicate"]
    assert len(client.calls) == 1
    assert agent.last_steps == 3
    assert llm.generate_calls == grounding.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("options,content,reason,model_calls", [
    ({"max_steps": 1}, "Je vais chercher ORION.", "step_limit", 2),
    ({}, "Texte sans observation. " * 5000, "context_budget", 2),
], ids=["iteration_limit", "context_limit"])
async def test_reminder_preserves_iteration_and_context_limits(monkeypatch, options, content, reason, model_calls):
    agent, llm, grounding, client = make_agent(monkeypatch, [message(evidence_plan()),
        {"role": "assistant", "content": content}, call("search_code", query="ORION")], **options)
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == reason
    assert agent.last_execution["completed"] == []
    assert agent.last_execution["remaining"] == ["1", "2"]
    assert len(llm.calls) == model_calls
    assert agent.last_steps <= options.get("max_steps", agent.max_steps)
    assert client.calls == llm.generate_calls == grounding.calls == []
    assert agent.memory.recent[-1]["observations"] == []
    reminders = [event for event in agent.last_execution["events"] if event["event"] == "tool_reminder"]
    assert len(reminders) == (0 if reason == "step_limit" else 1)


@pytest.mark.anyio
async def test_reminder_does_not_extend_tool_budget(monkeypatch):
    agent, llm, grounding, client = make_agent(monkeypatch, [message(evidence_plan()),
        call("search_code", query="ORION"), {"role": "assistant", "content": "Je vais lire ORION."},
        call("read_file", path="current.py")], [search_result()], max_tool_calls=1)
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "global_budget"
    assert agent.last_execution["completed"] == ["1"]
    assert [trace["status"] for trace in agent.last_trace] == ["success"]
    assert len(client.calls) == 1
    assert agent.last_steps == 2
    assert len(llm.calls) == 3
    assert not any(event["event"] == "tool_reminder" for event in agent.last_execution["events"])
    assert llm.generate_calls == grounding.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("read,accepted", [
    ({"path": "unit.py", "start_line": 4, "end_line": 5, "content": "def secondary():\n    return 2"}, False),
    ({"path": "unit.py", "start_line": 1, "end_line": 1, "content": "def primary():"}, False),
    ({"path": "unit.py", "start_line": 1, "end_line": 2, "content": "def primary():\n    return 1",
      "complete_symbols": [{"name": "primary", "start_line": 1, "end_line": 2}]}, True),
    ({"path": "different.py", "start_line": 1, "end_line": 2, "content": "def primary():\n    return 1"}, False),
    ({"path": "unit.py", "start_line": 1, "end_line": 2, "content": "def primary():\n    if ready:"}, False),
    ({"path": "unit.py", "start_line": 1, "end_line": 5, "content": "def primary():\n    return 1\n\ndef secondary():\n    return 2",
      "complete_symbols": [{"name": "primary", "start_line": 1, "end_line": 2}, {"name": "secondary", "start_line": 4, "end_line": 5}]}, True),
    ({"path": "unit.py", "start_line": 1, "end_line": 2, "content": "def primary():\n    value = 1", "complete_symbols": []}, False),
])
@pytest.mark.parametrize("query", ["primary", "def primary("])
async def test_R2_read_must_match_discovered_symbol(monkeypatch, read, accepted, query):
    question = "Localise primary, puis lis son implémentation."
    data = plan(question, ["Localise primary", "lis son implémentation"], [
        {"tool": "search_code", "kind": "contains", "value": "primary"},
        {"tool": "read_file", "kind": "nonempty", "from_step": "1"},
    ])
    agent, _, _, _ = make_agent(monkeypatch, [message(data), call("search_code", query=query),
        call("read_file", path="unit.py", start_line=read["start_line"], end_line=read["end_line"])], [
        tool_result(content={"matches": [{"file": "unit.py", "line": 1, "text": "def primary():"}]}),
        tool_result(content=read),
    ], generated=["primary est déclaré [S1]. secondary renvoie 2 [S2]." if not accepted else "primary renvoie 1 [S1]."], max_replans=0)
    await agent.ask(question)
    assert ("2" in agent.last_execution["completed"]) is accepted
    assert (agent.last_execution["stop_reason"] == "goal_achieved") is accepted


@pytest.mark.anyio
async def test_R2_general_file_read_need_not_match_a_symbol(monkeypatch):
    question = "Localise primary, puis lis le fichier retrouvé."
    data = plan(question, ["Localise primary", "lis le fichier retrouvé"], [
        {"tool": "search_code", "kind": "contains", "value": "primary"},
        {"tool": "read_file", "kind": "nonempty", "from_step": "1"},
    ])
    agent, _, _, _ = make_agent(monkeypatch, [message(data), call("search_code", query="primary"),
        call("read_file", path="unit.py", start_line=4, end_line=5)], [
        tool_result(content={"matches": [{"file": "unit.py", "line": 1, "text": "def primary():"}]}),
        tool_result(content={"path": "unit.py", "start_line": 4, "end_line": 5, "content": "def secondary():\n    return 2"}),
    ], generated=["primary est déclaré [S1]. secondary renvoie 2 [S2]."])
    await agent.ask(question)
    assert agent.last_execution["stop_reason"] == "goal_achieved"


@pytest.mark.anyio
@pytest.mark.parametrize("candidate,accepted", [
    ("alpha.py contient CAP [S1]. beta.py contient CAP [S2].", False),
    ("CAP vaut 7 dans alpha.py [S1], contre 13 dans beta.py [S2].", True),
])
async def test_R1_comparison_requires_values(monkeypatch, candidate, accepted):
    question = "Lis alpha.py, puis compare CAP avec beta.py."
    data = plan(question, ["Lis alpha.py", "compare CAP avec beta.py"], [{"tool": "read_file", "kind": "nonempty"}] * 2)
    agent, _, _, _ = make_agent(monkeypatch, [message(data), call("read_file", path="alpha.py"), call("read_file", path="beta.py")], [
        tool_result(content={"path": "alpha.py", "content": "CAP = 7"}),
        tool_result(content={"path": "beta.py", "content": "CAP = 13"}),
    ], generated=[candidate])
    await agent.ask(question)
    assert (agent.last_execution["stop_reason"] == "goal_achieved") is accepted


@pytest.mark.anyio
@pytest.mark.parametrize("left,right,candidate,accepted", [
    (7, 7, "CAP vaut 7 dans alpha.py et beta.py [S1][S2].", True),
    (-4, 28, "Dans alpha.py, CAP vaut -4 [S1].\n\nDans beta.py, CAP vaut 28 [S2].", True),
    (7, 13, "CAP vaut 7 dans alpha.py [S1]. CAP existe aussi dans beta.py [S2].", False),
    (7, 13, "CAP vaut 17 dans alpha.py [S1] et 13 dans beta.py [S2].", False),
    (7, 13, "CAP apparaît aux lignes 7 dans alpha.py [S1] et 13 dans beta.py [S2].", False),
    ("cold", "warm", 'CAP vaut "cold" dans alpha.py [S1] et "warm" dans beta.py [S2].', True),
    (None, 13, "CAP vaut 7 dans alpha.py [S1] et 13 dans beta.py [S2].", False),
])
async def test_R1_comparison_variants(monkeypatch, left, right, candidate, accepted):
    question = "Lis alpha.py, puis compare CAP avec beta.py."
    data = plan(question, ["Lis alpha.py", "compare CAP avec beta.py"], [{"tool": "read_file", "kind": "nonempty"}] * 2)
    agent, _, _, _ = make_agent(monkeypatch, [message(data), call("read_file", path="alpha.py"), call("read_file", path="beta.py")], [
        tool_result(content={"path": "alpha.py", "content": f"CAP = {left!r}" if left is not None else "# observation without the requested value"}),
        tool_result(content={"path": "beta.py", "content": f"CAP = {right!r}"}),
    ], generated=[candidate])
    await agent.ask(question)
    assert (agent.last_execution["stop_reason"] == "goal_achieved") is accepted


@pytest.mark.anyio
@pytest.mark.parametrize("compare,code,candidate,accepted", [
    (False, "CAP = calculate()", "alpha.py contient CAP [S1]. beta.py contient CAP [S2].", True),
    (True, "CAP = calculate()", "alpha.py contient CAP [S1]. beta.py contient CAP [S2].", False),
    (True, "CAP = 7\nCAP = calculate()", "CAP vaut 7 dans alpha.py [S1] et beta.py [S2].", False),
])
async def test_R1_general_read_and_uncertifiable_comparison(monkeypatch, compare, code, candidate, accepted):
    second = "compare CAP avec beta.py" if compare else "lis beta.py"
    question = f"Lis alpha.py, puis {second}."
    data = plan(question, ["Lis alpha.py", second], [{"tool": "read_file", "kind": "nonempty"}] * 2)
    agent, _, _, _ = make_agent(monkeypatch, [message(data), call("read_file", path="alpha.py"), call("read_file", path="beta.py")], [
        tool_result(content={"path": "alpha.py", "content": code}),
        tool_result(content={"path": "beta.py", "content": code}),
    ], generated=[candidate])
    await agent.ask(question)
    assert (agent.last_execution["stop_reason"] == "goal_achieved") is accepted


@pytest.mark.anyio
@pytest.mark.parametrize("prefix,candidate,accepted", [
    ("", "Implémentation [S1] :\n\n```python\ndef primary():\n    return 1\n```", False),
    ("Montre le code. ", "Implémentation [S1] :\n\n```python\ndef primary():\n    return 1\n```", True),
    ("", "primary renvoie 1 [S1].", True),
])
async def test_C2_output_contract_is_required_for_completion(monkeypatch, prefix, candidate, accepted):
    question = prefix + "Localise primary, puis lis son implémentation."
    data = plan(question, ["Localise primary", "lis son implémentation"], [
        {"tool": "search_code", "kind": "contains", "value": "primary"},
        {"tool": "read_file", "kind": "nonempty", "from_step": "1"},
    ])
    agent, _, _, _ = make_agent(monkeypatch, [message(data), call("search_code", query="primary"), call("read_file", path="unit.py")], [
        tool_result(content={"matches": [{"file": "unit.py", "line": 1, "text": "def primary():"}]}),
        tool_result(content={"path": "unit.py", "start_line": 1, "end_line": 2, "content": "def primary():\n    return 1",
                             "complete_symbols": [{"name": "primary", "start_line": 1, "end_line": 2}]}),
    ], generated=[candidate])
    await agent.ask(question)
    assert (agent.last_execution["stop_reason"] == "goal_achieved") is accepted


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["définition", "test associé"])
async def test_R4_observed_target_selected_by_objective(tmp_path, monkeypatch, target):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "model.py").write_text("class Widget:\n    pass\n", encoding="utf8")
    (tmp_path / "test_widget.py").write_text("from model import Widget\ndef test_widget():\n    assert Widget()\n", encoding="utf8")
    goal = f"Localise Widget, puis lis sa {target}."
    data = plan(goal, ["Localise Widget", f"lis sa {target}"], [
        {"tool": "search_code", "kind": "contains", "value": "Widget"},
        {"tool": "read_file", "kind": "nonempty", "from_step": "1"}])
    path = "model.py" if target == "définition" else "test_widget.py"
    answer = "Widget est déclaré dans model.py [S1][S2]." if target == "définition" else "Le test contient assert Widget() [S2]. Widget est déclaré dans model.py [S1]."
    llm = FakeLLM([message(data), call("search_code", query="Widget"), call("read_file", path=path)], generated=[answer])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp)
    result = await agent.ask(goal)
    assert agent.last_execution["stop_reason"] == "goal_achieved", result
    assert [t["status"] for t in agent.last_trace] == ["success", "success"]
    assert agent.last_trace[-1]["arguments"]["path"] == path
    assert "Widget" in agent._final_sources[-1]["content"]


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["transient", "permanent", "timeout", "exhausted"])
async def test_R3_real_mcp_identical_retry(tmp_path, mode):
    import asyncio
    from mcp.server.mcpserver.exceptions import ToolError
    attempts = []
    server = MCPServer("retry fixture")

    @server.tool()
    async def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        attempts.append(path)
        if mode == "timeout" and len(attempts) == 1:
            await asyncio.sleep(0.1)
        if mode == "permanent":
            raise ToolError("Path does not exist: a.py")
        if mode == "exhausted" or (mode == "transient" and len(attempts) == 1):
            raise ValueError("Transient backend failure")
        return {"path": path, "start_line": 1, "end_line": 1, "content": "VALUE = 8"}

    goal = "Lis a.py, puis lis b.py."
    data = plan(goal, ["Lis a.py", "lis b.py"], [{"tool": "read_file", "kind": "nonempty"}] * 2)
    llm = FakeLLM([message(data), call("read_file", path="a.py"), call("read_file", path="a.py"),
                   call("read_file", path="a.py"), call("read_file", path="b.py")],
                  generated=["a.py contient VALUE = 8 [S1]. b.py contient VALUE = 8 [S2]."])
    # On success, move directly to the second requested file.
    if mode in {"transient", "timeout"}:
        llm.responses.pop(3)
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=server,
                     max_replans=0, max_retries=1, tool_timeout=0.02, max_steps=8)
    await agent.ask(goal)
    if mode in {"transient", "timeout"}:
        assert attempts == ["a.py", "a.py", "b.py"]
        assert agent.last_execution["stop_reason"] == "goal_achieved"
    else:
        assert attempts == ["a.py"] * (1 if mode == "permanent" else 2)
        assert agent.last_execution["stop_reason"] != "goal_achieved"
    assert agent.execution.tool_calls <= 3


@pytest.mark.anyio
@pytest.mark.parametrize("missing,partial", [(None, False), ("a.py", False), ("b.py", False), (None, True)])
async def test_C3_real_mcp_comparison_needs_both_files_and_final_values(tmp_path, monkeypatch, missing, partial):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    for name, number in [("a.py", 10), ("b.py", 20)]:
        if name != missing:
            (tmp_path / name).write_text(f"# marker\nLIMIT = {number}\n", encoding="utf8")
    goal, data = comparison_plan()
    candidate = "a.py et b.py contiennent LIMIT [S1][S2][S3]." if partial else (
        "marker apparaît dans les deux fichiers [S1]. LIMIT vaut 10 dans a.py [S2] et 20 dans b.py [S3].")
    memory = ConversationMemory(tmp_path / "memory", "comparison")
    llm = FakeLLM([message(data), call("search_code", query="marker"), call("read_file", path="a.py"),
                   call("read_file", path="b.py")], generated=[candidate])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp, memory=memory, max_replans=0)
    answer = await agent.ask(goal)
    assert (agent.last_execution["stop_reason"] == "goal_achieved") is (not missing and not partial)
    successful = [trace for trace in agent.last_trace if trace["status"] == "success"]
    assert len(successful) == (3 if not missing else (1 if missing == "a.py" else 2))
    if not missing:
        assert agent.last_execution["completed"] == ["1", "2", "3"]
        assert "LIMIT = 10" in agent._final_sources[1]["content"]
        assert "LIMIT = 20" in agent._final_sources[2]["content"]
        if not partial:
            assert "vaut 10" in answer and "20 dans b.py" in answer
            assert agent.last_grounding == "passed"
    restored = ConversationMemory(tmp_path / "memory", "comparison")
    assert restored.total_turns == 1
    assert ("LIMIT" in json.dumps(restored.recent, ensure_ascii=False)) is (missing != "a.py")


@pytest.mark.anyio
@pytest.mark.parametrize("name,payload,claim", [
    ("get_git_diff", {"diff": ""}, "Le diff Git observé est vide"),
    ("search_code", {"matches": [], "truncated": False}, "La recherche observée ne trouve aucune occurrence"),
    ("run_tests", {"passed": False, "timed_out": False, "returncode": 1, "stdout": "1 failed", "stderr": ""},
     "Les tests exécutés ont un code de sortie égal à 1"),
])
async def test_C1_real_mcp_negative_collection_and_final(tmp_path, monkeypatch, name, payload, claim):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "state.py").write_text("FLAG = 1\n", encoding="utf8")
    server = MCPServer("negative observations")
    # Actual local MCP transport with controlled negative observations.
    def observe() -> dict:
        return payload
    server.tool(name=name)(observe)
    server.tool()(project_tools.read_file)
    first = {"get_git_diff": "Vérifie les modifications", "search_code": "Cherche absent_marker",
             "run_tests": "Vérifie le résultat des tests"}[name]
    goal = first + ", puis lis state.py."
    data = plan(goal, [first, "lis state.py"], [
        {"tool": name, "kind": "observed"}, {"tool": "read_file", "kind": "nonempty"}])
    candidate = claim + " [S1]. FLAG vaut 1 [S2]."
    llm = FakeLLM([message(data), call(name), call("read_file", path="state.py")], generated=[candidate])
    judge = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=judge, server=server)
    answer = await agent.ask(goal)
    assert answer == candidate
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert [trace["status"] for trace in agent.last_trace] == ["success", "success"]
    assert json.loads(agent._final_sources[0]["content"].partition("Result:\n")[2]) == payload
    assert len(judge.calls) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["timeout", "error", "negative_result"])
async def test_R3_run_tests_never_reexecutes_after_uncertain_execution(monkeypatch, failure):
    import asyncio
    from mcp.server.mcpserver.exceptions import ToolError
    attempts = []
    server = MCPServer("uncertain test execution")
    @server.tool()
    async def run_tests(path: str = "tests") -> dict:
        attempts.append(path)
        if failure == "timeout":
            await asyncio.sleep(0.1)
        if failure == "error":
            raise ToolError("Transient backend failure")
        return {"passed": False, "timed_out": failure == "negative_result", "returncode": None, "stdout": ""}
    # M6 permits direct examination of duplicate policy without replanning requirements.
    llm = FakeLLM([call("run_tests"), call("run_tests"), {"role": "assistant", "content": "stop"}], generated=["Indéterminé."])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=server,
                     planning="off", tool_timeout=0.02, max_steps=3)
    await agent.ask("Exécute les tests.")
    assert attempts == ["tests"]
    assert [trace["status"] for trace in agent.last_trace] == [
        {"timeout": "timeout", "error": "error", "negative_result": "success"}[failure], "duplicate"]


def test_C3_wrong_returned_file_cannot_supply_selected_read():
    goal, data = comparison_plan()
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=FakeGroundingLLM())
    agent.last_execution = {"plan": validate_plan(data, goal, ["search_code", "read_file"], 6)}
    wrong = build_tool_source("read_file", {"path": "a.py"}, json.dumps({"path": "b.py", "content": "LIMIT = 20"}))
    assert not agent._verifies_step(agent.last_execution["plan"][1], wrong, [wrong])


@pytest.mark.anyio
async def test_C1_real_project_tools_collect_empty_git_and_failed_pytest(tmp_path, monkeypatch):
    import subprocess
    from mcp.client import Client
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "test_failure.py").write_text("def test_failure():\n    assert 1 == 2\n", encoding="utf8")
    async with Client(mcp) as client:
        diff = await client.call_tool("get_git_diff", {})
        search = await client.call_tool("search_code", {"query": "missing_identifier"})
        tests = await client.call_tool("run_tests", {"path": "test_failure.py"})
    from app.agent.tool_loop import tool_result_content
    for name, result in [("get_git_diff", diff), ("search_code", search), ("run_tests", tests)]:
        assert not result.is_error
        assert verifies({"tool": name, "kind": "observed"}, name, tool_result_content(result))
        assert not verifies({"tool": name, "kind": "nonempty"}, name, tool_result_content(result))
    assert json.loads(tool_result_content(diff))["diff"] == ""
    data = json.loads(tool_result_content(tests))
    assert data["returncode"] == 1 and data["passed"] is False
    assert "assert 1 == 2" in data["stdout"] and "1 failed" in data["stdout"]


@pytest.mark.anyio
async def test_R5_R6_real_mcp_code_markdown_and_literal_survive_finalization(tmp_path, monkeypatch):
    from markdown_it import MarkdownIt
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    code = 'def literal():\n    return "[S2]"'
    (tmp_path / "literal.py").write_text(code + "\n", encoding="utf8")
    goal = "Montre le code. Localise literal, puis lis son implémentation."
    data = plan(goal, ["Localise literal", "lis son implémentation"], [
        {"tool": "search_code", "kind": "contains", "value": "literal"},
        {"tool": "read_file", "kind": "nonempty", "from_step": "1"}])
    candidate = f"Implémentation :\n\n```python\n{code}\n```"
    llm = FakeLLM([message(data), call("search_code", query="literal"), call("read_file", path="literal.py")],
                  generated=[candidate])
    judge = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=judge, server=mcp)
    answer = await agent.ask(goal)
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert [trace["status"] for trace in agent.last_trace] == ["success", "success"]
    assert agent.last_execution["completed"] == ["1", "2"]
    fences = [token.content for token in MarkdownIt().parse(answer) if token.type == "fence"]
    assert fences == [code + "\n"]
    compile(fences[0], "observed-literal", "exec")
    assert all('return "[S2]"' in item["prompt"] for item in judge.calls)
    assert answer.rstrip().endswith("[S1]")
