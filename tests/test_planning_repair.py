"""A rejected planner candidate may be repaired once, within existing budgets."""

import json

import pytest

from tests.test_planning import GOAL, evidence_plan
from tests.test_planning_integration import (
    call,
    make_agent,
    message,
    read_result,
    search_result,
)


OBJECTIVE_ERROR = "Step objective must be an exact excerpt of the user request."


def invalid_candidate():
    plan = evidence_plan()
    # This extra objective caused the recorded H plan rejection on 2026-10-15.
    plan["steps"].append({
        "id": "3", "objective": "lis les tests associés", "depends_on": ["2"],
        "success": {"tool": "read_file", "kind": "nonempty", "from_step": "2"},
    })
    return message(plan)


@pytest.mark.anyio
async def test_rejected_candidate_receives_exact_feedback_before_any_mcp_call(monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    rejected = invalid_candidate()
    accepted = message(evidence_plan())
    agent, llm, _, client = make_agent(monkeypatch, [rejected, accepted,
        call("search_code", query="ORION"), call("read_file", path="current.py")],
        [search_result(), read_result()], generated=["ORION vaut 42 [S1]."])
    original_chat = llm.chat

    def checked_chat(messages, tools=None):
        if len(llm.calls) == 1:
            assert client.calls == []
            assert agent.last_execution["plan"] == []
        return original_chat(messages, tools)

    monkeypatch.setattr(llm, "chat", checked_chat)
    assert await agent.ask(GOAL) == "ORION vaut 42 [S1]."
    feedback = "\n".join(item.get("content", "") for item in llm.calls[1]["messages"])
    assert rejected["content"] in feedback
    assert OBJECTIVE_ERROR in feedback
    assert llm.calls[0]["tools"] is llm.calls[1]["tools"] is None
    assert agent.last_debug["planner_responses"] == [rejected, accepted]
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["replans"] == 1
    assert agent.last_execution["completed"] == ["1", "2"]
    assert len(client.calls) == 2
    assert agent.last_stage_stats["planner"] == {"prompt_tokens": 20, "completion_tokens": 4}


@pytest.mark.anyio
async def test_second_invalid_candidate_stops_without_third_attempt(monkeypatch):
    agent, llm, _, client = make_agent(monkeypatch,
        [invalid_candidate(), invalid_candidate(), message(evidence_plan())])
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "invalid_plan"
    assert agent.last_execution["plan"] == []
    assert agent.last_execution["replans"] == 1
    assert len(llm.calls) == 2
    assert client.calls == llm.generate_calls == []


@pytest.mark.anyio
async def test_repair_requires_remaining_replan_allowance(monkeypatch):
    agent, llm, _, client = make_agent(monkeypatch,
        [invalid_candidate(), message(evidence_plan())], max_replans=0)
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "invalid_plan"
    assert agent.last_execution["replans"] == 0
    assert len(llm.calls) == 1
    assert client.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("budget,reason", [
    ("tokens", "global_budget"),
    ("seconds", "global_budget"),
    ("context", "context_budget"),
])
async def test_repair_checks_remaining_budgets_before_second_model_call(monkeypatch, budget, reason):
    rejected = invalid_candidate()
    if budget == "context":
        data = json.loads(rejected["content"])
        data["untrusted_extra"] = "x" * 20_000
        rejected = message(data)
    agent, llm, _, client = make_agent(monkeypatch, [rejected, message(evidence_plan())])
    original_chat = llm.chat

    def deplete_budget(messages, tools=None):
        response = original_chat(messages, tools)
        if budget == "tokens":
            agent.execution.used_tokens = agent.execution.max_total_tokens
        elif budget == "seconds":
            agent.execution.started -= agent.execution.max_seconds + 1
        return response

    monkeypatch.setattr(llm, "chat", deplete_budget)
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == reason
    assert agent.last_execution["plan"] == []
    assert len(llm.calls) == 1
    assert client.calls == llm.generate_calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["user_decision", "transport"])
async def test_user_decision_and_transport_failure_do_not_trigger_repair(monkeypatch, failure):
    decision = message({"stop": "user_decision", "reason": "Quel projet faut-il analyser ?"})
    agent, llm, _, client = make_agent(monkeypatch, [decision, message(evidence_plan())])
    if failure == "transport":
        def broken_chat(messages, tools=None):
            llm.calls.append({"messages": messages, "tools": tools})
            raise OSError("connection reset")
        monkeypatch.setattr(llm, "chat", broken_chat)
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == ("execution_error" if failure == "transport" else "user_decision")
    assert agent.last_execution["replans"] == 0
    assert len(llm.calls) == 1
    assert client.calls == []


@pytest.mark.anyio
async def test_repaired_replan_preserves_completed_evidence_and_shared_allowance(monkeypatch):
    rewritten = evidence_plan()
    rewritten["steps"][0]["success"]["kind"] = "nonempty"
    agent, llm, _, client = make_agent(monkeypatch, [message(evidence_plan()),
        call("search_code", query="ORION"), call("read_file", path="current.py"),
        message(rewritten), message(evidence_plan()),
        call("read_file", path="current.py", start_line=1, end_line=2)],
        [search_result(), read_result(content="OTHER = 1"), read_result(content="ORION = 42")],
        generated=["ORION vaut 42 [S2]."], max_steps=5)
    await agent.ask(GOAL)
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert agent.last_execution["replans"] == 2
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_execution["plan"][0]["success"] == evidence_plan()["steps"][0]["success"]
    assert agent.last_execution["plan"][0]["evidence"] == ["S1"]
    assert [name for name, _ in client.calls] == ["search_code", "read_file", "read_file"]
    feedback = "\n".join(item.get("content", "") for item in llm.calls[4]["messages"])
    assert "Replanning cannot change completed steps." in feedback
