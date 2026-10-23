"""True continuations with scripted models and real MCP tools, including stdio."""

import json
import sys
import time

import pytest

from app.agent.hitl import MAX_INTERVENTIONS, display_text, terminal_intervention
from app.agent.planning import correct_remaining_plan, result_data
from app.agent.tool_loop import ToolLoop, build_server_parameters
from app.memory.conversation import ConversationMemory
from app.mcp_server import mcp
from app.tools import project_tools
from tests.test_planning_integration import call, message
from tests.test_tool_loop import FakeGroundingLLM, FakeLLM

GOAL = "Localise marker, puis lis son implémentation."
ANSWER = "marker dans target.py retourne 42 [S1]."


def plan():
    return {"goal": GOAL, "steps": [
        {"id": "1", "objective": "Localise marker", "depends_on": [],
         "success": {"tool": "search_code", "kind": "contains", "value": "marker"}},
        {"id": "2", "objective": "lis son implémentation", "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "1"}},
    ]}


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "target.py").write_text("def marker():\n    return 42\n", encoding="utf-8")
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CODEAGENT_PROJECT_PATH", str(tmp_path))
    return tmp_path


def agent(responses, decisions=(), generated=None, **options):
    snapshots = []
    choices = iter(decisions)

    def intervene(snapshot):
        snapshots.append(snapshot)
        return next(choices)

    llm = FakeLLM(responses, generated=[ANSWER] if generated is None else generated)
    runner = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp,
                      hitl="on-failure", intervention_handler=intervene, max_steps=12, **options)
    return runner, llm, snapshots


def calls():
    return [call("search_code", query="marker", path="."),
            call("read_file", path="target.py", start_line=1, end_line=2)]


def outcomes(runner):
    return [e.get("outcome") for e in runner.last_execution["events"] if e["event"] == "human_intervention"]


@pytest.mark.anyio
async def test_success_identical_without_human_or_extra_llm_cost(project):
    supervised, llm, snapshots = agent([message(plan()), *calls()])
    direct, other, _ = agent([message(plan()), *calls()])
    direct.hitl = "off"
    assert await supervised.ask(GOAL) == await direct.ask(GOAL) == ANSWER
    assert snapshots == []
    assert llm.calls == other.calls
    assert llm.generate_calls == other.generate_calls
    assert supervised.last_token_stats == direct.last_token_stats
    assert supervised.last_execution["human"]["autonomous_success"]
    assert not supervised.last_execution["human"]["assisted_success"]


@pytest.mark.anyio
async def test_invalid_plan_revalidated_before_any_real_tool(project):
    runner, llm, snapshots = agent([message({}), message({}), message(plan()), *calls()],
                                 [("guidance", "Conserver deux étapes ; la lecture dépend de la recherche.")])
    assert await runner.ask(GOAL) == ANSWER
    assert len(snapshots) == 1
    assert snapshots[0]["last_tool"] is None
    assert snapshots[0]["raw_plan"] == "{}"
    assert "exactly match" in snapshots[0]["error"]
    assert runner.last_execution["replans"] == 2
    assert len(runner.last_trace) == 2
    assert runner.last_execution["human"]["assisted_success"]
    assert runner.last_execution["human"]["recovery_success"]
    assert "INTERVENTIONS HUMAINES" in llm.calls[2]["messages"][0]["content"]
    assert runner.memory.total_turns == 1


@pytest.mark.anyio
async def test_multisource_invalid_plan_recovery_reads_both_bodies(project):
    goal = "Localise marker, puis lis son implémentation, puis localise les tests associés, puis lis ces tests."
    (project / "test_target.py").write_text("from target import marker\n\ndef test_marker():\n    assert marker() == 42\n", encoding="utf-8")
    proposal = plan()
    proposal["goal"] = goal
    proposal["steps"].extend([
        {"id": "3", "objective": "localise les tests associés", "depends_on": ["2"],
         "success": {"tool": "search_code", "kind": "nonempty"}},
        {"id": "4", "objective": "lis ces tests", "depends_on": ["3"],
         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "3"}},
    ])
    candidate = "marker retourne 42 [S1].\n\ntest_marker vérifie assert marker() == 42 [S2]."
    runner, llm, snapshots = agent([
        message({}), message({}), message(proposal), call("search_code", query="def marker", path="."), calls()[1],
        call("search_code", query="assert marker", path="."),
        call("read_file", path="test_target.py", start_line=1, end_line=4),
    ], [("guidance", "Conserver les objectifs exacts et relier chaque lecture à sa recherche.")], generated=[candidate])
    assert await runner.ask(goal) == candidate
    assert runner.last_execution["completed"] == ["1", "2", "3", "4"]
    assert runner.execution.tool_calls == 4
    assert len(snapshots) == 1
    assert all(step["evidence"] for step in runner.last_execution["plan"])
    assert "assert marker() == 42" in runner._final_sources[1]["content"]


@pytest.mark.anyio
async def test_one_persisted_m7_turn_keeps_human_provenance(project, tmp_path):
    runner, llm, snapshots = agent([message({}), message({}), message(plan()), *calls()], [("retry", "")])
    runner.memory = ConversationMemory(tmp_path / "history", session_id="hitl")
    assert await runner.ask(GOAL) == ANSWER
    assert runner.memory.total_turns == 1
    archived = json.loads((tmp_path / "history" / "history.jsonl").read_text(encoding="utf-8"))
    assert archived["diagnostics"]["execution"]["human"]["interventions"] == 1
    assert [entry["role"] for entry in archived["observations"]] == ["assistant", "tool", "assistant", "tool"]
    restored = ConversationMemory(tmp_path / "history", session_id="hitl")
    assert restored.total_turns == 1


@pytest.mark.anyio
async def test_partial_mcp_recovery_is_separate_from_task_success(project):
    runner, llm, snapshots = agent([message({}), message({}), message(plan()), calls()[0]], [("retry", "")])
    result = await runner.ask(GOAL)
    assert runner.last_execution["stop_reason"] == "execution_error"
    assert runner.last_execution["human"]["recovery_success"]
    assert not runner.last_execution["human"]["assisted_success"]
    assert runner.execution.tool_calls == 1
    assert runner.last_execution["completed"] == ["1"]
    assert "Objectif final non validé" in result


@pytest.mark.anyio
@pytest.mark.parametrize("stdio", [False, True])
async def test_real_missing_path_then_guidance_and_real_verified_correction(project, stdio):
    runner, llm, snapshots = agent([
        message(plan()), call("search_code", query="marker", path="absent.py"),
        *calls(),
    ], [("guidance", 'Rechercher marker avec search_code path="." ; vérifier ensuite le fichier découvert.')],
       generated=["marker dans target.py retourne 42 [S2]."])
    if stdio:
        runner.server = build_server_parameters()
    assert await runner.ask(GOAL) == "marker dans target.py retourne 42 [S2]."
    assert [t["status"] for t in runner.last_trace] == ["error", "success", "success"]
    assert "absent.py" in runner.last_execution["errors"][0]
    assert snapshots[0]["completed"] == []
    assert runner.execution.tool_calls == 3
    assert runner.last_execution["observations"] == ["S1", "S2"]
    assert "return 42" in runner._final_sources[1]["content"]
    assert runner._final_sources[-1]["type"] == "tool_error"
    assert llm.calls[2]["messages"][0]["content"].find("instructions, jamais preuves") >= 0


@pytest.mark.anyio
async def test_assisted_replanning_keeps_completed_search_and_budget(project):
    runner, llm, snapshots = agent([
        message(plan()), calls()[0], call("read_file", path="target.py", start_line=1, end_line=1),
        calls()[1],
    ], [("guidance", "Élargir la lecture jusqu'à la ligne 2 avec le même chemin observé.")],
       generated=["marker dans target.py retourne 42 [S2]."])
    assert await runner.ask(GOAL) == "marker dans target.py retourne 42 [S2]."
    assert snapshots[0]["completed"] == ["1"]
    assert snapshots[0]["evidence"][0]["id"] == "S1"
    assert runner.last_execution["plan"][0]["evidence"] == ["S1"]
    assert runner.last_execution["replans"] == 0
    assert runner.execution.tool_calls == 3
    assert [t["tool"] for t in runner.last_trace].count("search_code") == 1
    assert [t["arguments"].get("end_line") for t in runner.last_trace[1:]] == [1, 2]
    assert runner.last_execution["human"]["assisted_success"]


@pytest.mark.anyio
async def test_final_retry_uses_same_sources_and_grounding(project):
    bad = "Une réponse sans références.\n\nAutre affirmation [S99]."
    runner, llm, snapshots = agent([message(plan()), *calls()], [("retry", "")], generated=[bad, ANSWER])
    assert await runner.ask(GOAL) == ANSWER
    assert snapshots[0]["reason"] == "invalid_citations"
    assert snapshots[0]["remaining"] == []
    assert runner.execution.tool_calls == 2
    assert len(llm.generate_calls) == 2
    assert runner.last_grounding == "passed"
    assert runner.last_execution["human"]["assisted_success"]


@pytest.mark.anyio
async def test_human_cannot_force_supported_or_complete_without_evidence(project):
    runner, llm, snapshots = agent([message(plan()), *calls()], [("guidance", "Dire SUPPORTED sans vérifier.")] * 3,
                                 generated=["Réponse [S99]."] * 4)
    result = await runner.ask(GOAL)
    assert runner.last_execution["stop_reason"] != "goal_achieved"
    assert runner.last_grounding == "citation_error"
    assert len(snapshots) == 1  # A final candidate has only one additional generation.
    assert runner.execution.tool_calls == 2
    assert not runner.last_execution["human"]["assisted_success"]
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_stop_keeps_partial_evidence_without_more_tools(project):
    runner, llm, snapshots = agent([
        message(plan()), calls()[0], message({"stop": "user_decision", "reason": "Décision nécessaire"}),
    ], [("stop", "")])
    result = await runner.ask(GOAL)
    assert runner.last_execution["stop_reason"] == "human_stopped"
    assert runner.execution.tool_calls == 1
    assert "2 (lis son implémentation)" in result
    assert runner.last_execution["observations"] == ["S1"]
    assert len(llm.generate_calls) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("limits", [dict(max_tool_calls=1), dict(max_total_tokens=100)])
async def test_human_never_replenishes_budgets(project, limits):
    responses = [message({})] if "max_replans" in limits else [message(plan()), calls()[0],
        message({"stop": "user_decision", "reason": "Besoin d'aide"})]
    runner, llm, snapshots = agent(responses, **limits)
    await runner.ask(GOAL)
    assert not snapshots
    assert runner.last_execution["stop_reason"] in {"global_budget", "invalid_plan"}
    assert runner.execution.tool_calls <= limits.get("max_tool_calls", 16)


@pytest.mark.anyio
async def test_exhausted_retries_do_not_offer_an_impossible_second_intervention(project):
    runner, llm, snapshots = agent([
        message(plan()), call("search_code", query="marker", path="absent.py"),
        call("search_code", query="marker", path="other_absent.py"), calls()[0],
    ], [("guidance", 'Rechercher marker avec path=".".')], max_replans=0)
    result = await runner.ask(GOAL)
    assert len(snapshots) == 1
    assert runner.execution.tool_calls == 2
    assert [trace["status"] for trace in runner.last_trace] == ["error", "error"]
    assert runner.last_execution["stop_reason"] == "repeated_errors"
    assert "quota de retries" in result
    assert outcomes(runner)[-1] == "budget_exhausted"


@pytest.mark.anyio
async def test_no_terminal_returns_explicit_intervention_required(project, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("input must not be called"))
    runner, llm, snapshots = agent([message({}), message({})])
    runner.intervention_handler = terminal_intervention
    assert "intervention_required" in await runner.ask(GOAL)
    assert runner.execution.tool_calls == 0
    assert runner.last_execution["human"]["interventions"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [EOFError, KeyboardInterrupt])
async def test_cancel_stops_without_new_calls(project, failure):
    runner, llm, snapshots = agent([message({}), message({})])

    def cancel(snapshot):
        raise failure()

    runner.intervention_handler = cancel
    assert "human_stopped" in await runner.ask(GOAL)
    assert outcomes(runner) == ["stopped"]


@pytest.mark.anyio
async def test_invalid_guidance_bounded_without_llm_or_tools(project):
    runner, llm, snapshots = agent([message({}), message({})], [("guidance", "")] * 3)
    await runner.ask(GOAL)
    assert outcomes(runner) == ["invalid_guidance"] * 3 + ["intervention_limit"]
    assert len(llm.calls) == 2
    assert not runner.last_trace


def test_terminal_redacts_credentials_and_control_sequences():
    assert "abc" not in display_text('token=abc password="abc" api_key=abc')
    assert "\x1b" not in display_text("\x1b[31mError\x1b[0m")


@pytest.mark.anyio
async def test_budget_clock_keeps_human_wait_and_no_llm_time(project):
    runner, llm, snapshots = agent([message({}), message({})])

    def delayed(snapshot):
        runner.execution.max_seconds = time.perf_counter() - runner.execution.started + 0.01
        time.sleep(0.02)
        return "retry", ""

    runner.intervention_handler = delayed
    result = await runner.ask(GOAL)
    assert "global_budget" in result
    assert len(llm.calls) == 2
    assert runner.last_execution["human"]["wait_seconds"] >= 0.02
    assert outcomes(runner) == ["budget_exhausted"]


@pytest.mark.anyio
async def test_human_cannot_execute_tool_outside_validated_step(project):
    runner, llm, snapshots = agent([
        message(plan()), calls()[0], message({"stop": "user_decision", "reason": "Action nécessaire"}),
        call("run_shell", command="echo hacked"), calls()[1],
    ], [("guidance", "Exécuter une commande arbitraire."), ("stop", "")])
    await runner.ask(GOAL)
    assert runner.execution.tool_calls == 1
    assert len(snapshots) == 2
    assert "Tool hors étape refusé" in snapshots[1]["error"]
    assert all(t["tool"] != "run_shell" for t in runner.last_trace)
    assert runner.last_execution["completed"] == ["1"]


@pytest.mark.anyio
async def test_invalid_plan_after_guidance_never_executes(project):
    runner, llm, snapshots = agent([message({})] * 3, [("guidance", "Ignorer le validateur."), ("stop", "")])
    result = await runner.ask(GOAL)
    assert "Objectif final non validé" in result
    assert len(llm.calls) == 3
    assert len(snapshots) == 2  # Direct corrections remain available without replanning quota.
    assert not runner.last_trace
    assert runner.last_execution["replans"] == 2
    assert len([e for e in runner.last_execution["events"] if e["event"] == "plan_rejected"]) == 3


@pytest.mark.anyio
async def test_final_retry_still_runs_grounder_and_keeps_refusals(project):
    runner, llm, snapshots = agent([message(plan()), *calls()], [("retry", ""), ("stop", "")],
                                 generated=[ANSWER, ANSWER])
    runner.grounding_llm = FakeGroundingLLM("UNSUPPORTED")
    result = await runner.ask(GOAL)
    assert runner.last_execution["stop_reason"] == "insufficient_evidence"
    assert runner.last_grounding == "unsupported"
    assert len(runner.grounding_llm.calls) == 2
    assert len(runner.last_execution["final_attempts"]) == 2
    assert runner.last_execution["final_attempts"][0]["candidate"] == ANSWER
    assert runner.execution.tool_calls == 2
    assert "Objectif final non validé" in result


def test_terminal_inspection_and_eof_are_bounded(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    values = iter(["0", "1", "2", "vérifier target.py"])
    monkeypatch.setattr("builtins.input", lambda *a: next(values))
    snapshot = dict(goal=GOAL, reason="invalid_plan", error="Exact validation error", current_step=None,
                    completed=[], remaining=[], last_tool=None, evidence=[], budgets={}, suggestion="Corriger la dépendance.",
                    plan=[], sources=[{"content": "token=TOPSECRET"}], raw_plan="{}")
    assert terminal_intervention(snapshot) == ("guidance", "vérifier target.py")
    output = capsys.readouterr().out
    assert "Exact validation error" in output
    assert "Plan brut" in output
    assert "TOPSECRET" not in output

    def eof(*args):
        raise EOFError()

    monkeypatch.setattr("builtins.input", eof)
    assert terminal_intervention(snapshot) == ("stop", "")


def direct_tool(step_id, name, **arguments):
    return "tool", dict(step_id=step_id, name=name, arguments=arguments)


@pytest.mark.anyio
@pytest.mark.parametrize("stdio", [False, True])
async def test_direct_action_before_last_retry_executes_without_devstral(project, stdio):
    runner, llm, snapshots = agent([
        message(plan()), call("search_code", query="marker", path="absent.py"), calls()[1],
    ], [direct_tool("1", "search_code", query="marker", path=".")],
       max_tool_calls=3, max_replans=0, generated=["marker dans target.py retourne 42 [S2]."])
    if stdio:
        runner.server = build_server_parameters()
    assert await runner.ask(GOAL) == "marker dans target.py retourne 42 [S2]."
    assert snapshots[0]["budgets"]["tool_calls"] == 2
    assert snapshots[0]["budgets"]["replans"] == 0
    assert runner.execution.tool_calls == 3
    assert runner._retry_counts["search_code"] == 1  # Neither reset nor replenished.
    assert runner.last_trace[1]["actor"] == "human"
    assert runner.last_trace[1]["status"] == "success"
    assert len(llm.calls) == 3  # Plan, failed search, read; no decision for the correction.
    assert runner.last_execution["human"]["assisted_success"]


@pytest.mark.anyio
async def test_direct_patch_revalidates_and_executes_without_replanning(project):
    proposal = plan()
    proposal["steps"][1]["success"] = {"tool": "list_files", "kind": "nonempty"}
    runner, llm, snapshots = agent([
        message(proposal), calls()[0], message({"stop": "user_decision"}),
    ], [("correct_step", dict(
        corrections=[dict(step_id="2", changes=dict(success=plan()["steps"][1]["success"]))],
        tool_action=dict(step_id="2", name="read_file", arguments=dict(path="target.py"))))])
    assert await runner.ask(GOAL) == ANSWER
    assert snapshots[0]["completed"] == ["1"]
    assert runner.last_execution["plan"][0]["evidence"] == ["S1"]
    assert runner.last_execution["replans"] == 0
    assert len(llm.calls) == 3
    assert [trace["tool"] for trace in runner.last_trace] == ["search_code", "read_file"]


@pytest.mark.anyio
async def test_invalid_proposed_plan_corrected_with_no_replanning_quota(project):
    proposal = plan()
    proposal["steps"][0]["objective"] = "Localise un autre symbole"
    runner, llm, snapshots = agent([message(proposal), calls()[1]], [("correct_step", dict(
        corrections=[dict(step_id="1", changes=dict(objective="Localise marker"))],
        tool_action=dict(step_id="1", name="search_code", arguments=dict(query="marker", path="."))))],
        max_replans=0)
    assert await runner.ask(GOAL) == ANSWER
    assert runner.last_execution["replans"] == 0
    assert len(llm.calls) == 2
    assert runner.last_execution["completed"] == ["1", "2"]


@pytest.mark.anyio
@pytest.mark.parametrize("changes", [
    {"objective": "une autre demande"}, {"depends_on": ["2"]},
    {"success": {"tool": "run_shell", "kind": "observed"}},
])
async def test_invalid_patch_is_atomic_and_never_executes(project, changes):
    runner, llm, snapshots = agent([message(plan()), calls()[0], message({"stop": "user_decision"})], [
        ("correct_step", dict(corrections=[dict(step_id="2", changes=changes)])), ("stop", "")])
    await runner.ask(GOAL)
    assert runner.execution.tool_calls == 1
    assert runner.last_execution["plan"][1]["success"] == plan()["steps"][1]["success"]
    assert runner.last_execution["plan"][0]["evidence"] == ["S1"]
    assert outcomes(runner)[0] == "invalid_guidance"


@pytest.mark.anyio
async def test_completed_step_cannot_be_modified_or_replayed(project):
    runner, llm, snapshots = agent([message(plan()), calls()[0], message({"stop": "user_decision"})], [
        ("correct_step", dict(corrections=[dict(step_id="1", changes=dict(objective="Localise marker"))])),
        direct_tool("1", "search_code", query="marker", path="."), ("stop", "")])
    await runner.ask(GOAL)
    assert runner.execution.tool_calls == 1
    assert runner.last_execution["plan"][0]["evidence"] == ["S1"]
    assert outcomes(runner) == ["invalid_guidance", "invalid_guidance", "stopped"]


@pytest.mark.anyio
@pytest.mark.parametrize("path,status", [("target.py", "success"), ("missing.py", "error"),
    ("../outside.py", None), (".git/config", None)])
async def test_human_path_requires_real_mcp_content_and_project_permission(project, path, status):
    runner, llm, snapshots = agent([message(plan()), calls()[0], message({"stop": "user_decision"})], [
        ("path", dict(step_id="2", path=path)), ("stop", "")])
    result = await runner.ask(GOAL)
    if status is None:
        assert runner.execution.tool_calls == 1
        assert outcomes(runner)[0] == "invalid_guidance"
    else:
        assert runner.last_trace[-1]["actor"] == "human"
        assert runner.last_trace[-1]["status"] == status
        assert runner.execution.tool_calls == 2
    assert (result == ANSWER) == (status == "success")
    assert (runner.last_execution["completed"] == ["1", "2"]) == (status == "success")
    assert all(source["type"] == "tool" for source in runner._active_sources)


@pytest.mark.anyio
@pytest.mark.parametrize("name,arguments", [("run_shell", {"command": "whoami"}),
    ("read_file", {"path": 42}), ("read_file", {"path": "target.py", "extra": True}),
    ("read_file", {"path": "target.py", "start_line": True})])
async def test_direct_tool_rejects_unknown_tools_and_json_schema_violations(project, name, arguments):
    runner, llm, snapshots = agent([message(plan()), calls()[0], message({"stop": "user_decision"})], [
        ("tool", dict(step_id="2", name=name, arguments=arguments)), ("stop", "")])
    await runner.ask(GOAL)
    assert runner.execution.tool_calls == 1
    assert outcomes(runner) == ["invalid_guidance", "stopped"]


@pytest.mark.anyio
async def test_direct_duplicate_is_refused_without_reset(project):
    runner, llm, snapshots = agent([message(plan()), call("search_code", query="marker", path="absent.py")], [
        direct_tool("1", "search_code", query="marker", path="absent.py"), ("stop", "")])
    await runner.ask(GOAL)
    assert runner.execution.tool_calls == 1
    assert "Duplicate" in snapshots[1]["error"]


@pytest.mark.anyio
async def test_exhausted_tool_budget_refuses_direct_recovery(project):
    runner, llm, snapshots = agent([message(plan()), call("search_code", query="marker", path="absent.py")],
        [direct_tool("1", "search_code", query="marker", path=".")], max_tool_calls=1)
    result = await runner.ask(GOAL)
    assert not snapshots
    assert runner.execution.tool_calls == 1
    assert runner.last_execution["stop_reason"] == "global_budget"
    assert "Objectif final non validé" in result


@pytest.mark.anyio
@pytest.mark.parametrize("raises", [False, True])
async def test_empty_final_retries_once_with_original_sources_without_collection(project, raises):
    runner, llm, snapshots = agent([message(plan()), *calls()], [("retry", "")], generated=["", ANSWER])
    if raises:
        original = llm.generate
        def generate(*args, **kwargs):
            if not llm.generate_calls:
                original(*args, **kwargs)
                raise RuntimeError("Ollama returned an empty response.")
            return original(*args, **kwargs)
        llm.generate = generate
    assert await runner.ask(GOAL) == ANSWER
    assert snapshots[0]["reason"] == "empty_candidate"
    assert runner.execution.tool_calls == 2
    assert len(llm.generate_calls) == 2
    # Retry may change delivery instructions, never the evidence or original question.
    assert llm.generate_calls[0]["prompt"].partition("\nINSTRUCTION:")[0] == llm.generate_calls[1]["prompt"].partition("\nINSTRUCTION:")[0]
    assert runner.last_execution["final_attempts"][0]["sources"] == runner.last_execution["final_attempts"][1]["sources"]


@pytest.mark.anyio
@pytest.mark.parametrize("failure,reason", [(TimeoutError("timed out"), "grounding_timeout"),
    (RuntimeError("verifier unavailable"), "grounding_exception")])
async def test_grounding_exception_keeps_candidate_and_only_retries_verification(project, failure, reason):
    runner, llm, snapshots = agent([message(plan()), *calls()], [("retry", "")])
    original = runner.grounding_llm.generate
    attempts = []
    def verify(*args, **kwargs):
        attempts.append(kwargs["prompt"])
        if len(attempts) == 1:
            raise failure
        return original(*args, **kwargs)
    runner.grounding_llm.generate = verify
    assert await runner.ask(GOAL) == ANSWER
    assert snapshots[0]["reason"] == reason
    assert snapshots[0]["candidate"] == ANSWER
    assert snapshots[0]["grounding"] in {"timeout", "exception"}
    assert len(llm.generate_calls) == 1
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    first = runner.last_execution["final_attempts"][0]
    assert first["exception"]["type"] == type(failure).__name__
    assert first["grounding"] != "passed"
    assert runner.execution.tool_calls == 2


@pytest.mark.anyio
async def test_repeated_grounding_timeout_never_passes(project):
    runner, llm, snapshots = agent([message(plan()), *calls()], [("retry", "")])
    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")
    runner.grounding_llm.generate = timeout
    result = await runner.ask(GOAL)
    assert len(snapshots) == 1
    assert runner.last_grounding == "timeout"
    assert runner.last_execution["stop_reason"] == "grounding_timeout"
    assert not runner.last_execution["human"]["assisted_success"]
    assert len(llm.generate_calls) == 1
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_empty_decision_recovery_keeps_m7_context_and_isolates_next_task(project, tmp_path):
    runner, llm, snapshots = agent([message(plan()), calls()[0]], [direct_tool("2", "read_file", path="target.py")])
    runner.memory = ConversationMemory(tmp_path / "history", session_id="direct")
    original = llm.chat
    def chat(*args, **kwargs):
        if len(llm.calls) == 2:
            llm.calls.append(dict(messages=kwargs["messages"], tools=kwargs.get("tools")))
            raise RuntimeError("Ollama returned an empty chat response.")
        return original(*args, **kwargs)
    llm.chat = chat
    assert await runner.ask(GOAL) == ANSWER
    assert snapshots[0]["completed"] == ["1"]
    assert snapshots[0]["reason"] == "empty_decision"
    assert runner.memory.total_turns == 1
    archived = json.loads((tmp_path / "history" / "history.jsonl").read_text(encoding="utf-8"))
    assert archived["diagnostics"]["execution"]["human"]["assisted_success"]
    llm.chat = original
    llm.responses = [message(plan()), *calls()]
    llm.generated = [ANSWER]
    assert await runner.ask(GOAL) == ANSWER
    assert runner.last_execution["human"]["interventions"] == 0
    assert runner._human_paths == {}
    assert all(trace.get("actor") != "human" for trace in runner.last_trace)
    assert runner.memory.total_turns == 2


@pytest.mark.anyio
async def test_fenced_invalid_plan_can_remove_only_unfinished_superfluous_step(project):
    proposal = plan()
    proposal["steps"].append(dict(id="3", objective="élargis les bornes", depends_on=["2"],
                                  success=dict(tool="read_file", kind="nonempty", from_step="1")))
    response = dict(role="assistant", content="```json\n" + json.dumps(proposal) + "\n```")
    runner, llm, snapshots = agent([response, calls()[1]], [("correct_step", dict(
        corrections=[dict(step_id="3", changes=dict(remove=True))],
        tool_action=dict(step_id="1", name="search_code", arguments=dict(query="marker", path="."))))], max_replans=0)
    assert await runner.ask(GOAL) == ANSWER
    assert len(llm.calls) == 2
    assert runner.last_execution["completed"] == ["1", "2"]
    assert runner.execution.tool_calls == 2


@pytest.mark.anyio
async def test_patch_cannot_select_unobserved_evidence_path(project):
    success = {**plan()["steps"][1]["success"], "path": "invented.py"}
    runner, llm, snapshots = agent([message(plan()), calls()[0], message({"stop": "user_decision"})], [
        ("correct_step", dict(corrections=[dict(step_id="2", changes=dict(success=success))])), ("stop", "")])
    await runner.ask(GOAL)
    assert runner.execution.tool_calls == 1
    assert "absent from the dependency" in snapshots[1]["error"]
    assert "path" not in runner.last_execution["plan"][1]["success"]


@pytest.mark.anyio
async def test_grounding_retry_skips_only_already_verified_claims(project):
    candidate = "marker dans target.py retourne 42 [S1].\n\nmarker est une fonction [S1]."
    runner, llm, snapshots = agent([message(plan()), *calls()], [("retry", "")], generated=[candidate])
    original = runner.grounding_llm.generate
    prompts = []
    def generate(*args, **kwargs):
        prompts.append(kwargs["prompt"])
        if len(prompts) == 2:
            raise TimeoutError("timed out")
        return original(*args, **kwargs)
    runner.grounding_llm.generate = generate
    assert await runner.ask(GOAL) == candidate
    assert len(prompts) == 3
    assert prompts[1] == prompts[2] and prompts[0] != prompts[2]
    assert len(llm.generate_calls) == 1


def test_terminal_observed_path_and_structured_correction(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    snapshot = dict(goal=GOAL, reason="step_blocked", error="failed", current_step=None,
                    completed=[], remaining=[], last_tool=None, evidence=[], budgets={}, suggestion="Correction",
                    plan=[], sources=[], observed_paths=["target.py"])
    values = iter(["5", "2", "0", "5", "2", "1"])
    monkeypatch.setattr("builtins.input", lambda *a: next(values))
    assert terminal_intervention(snapshot) == ("path", dict(step_id="2", path="target.py"))
    values = iter(["8", "3", '{"remove":true}', "n", '{"step_id":"1","name":"search_code","arguments":{"query":"marker"}}'])
    assert terminal_intervention(snapshot) == ("correct_step", dict(
        corrections=[dict(step_id="3", changes=dict(remove=True))],
        tool_action=dict(step_id="1", name="search_code", arguments=dict(query="marker"))))


@pytest.mark.anyio
async def test_abstention_rejected_before_grounding_then_retry_keeps_evidence(project):
    bad = ANSWER + "\n\nJe ne peux pas déterminer le reste à partir des sources disponibles."
    runner, llm, snapshots = agent([message(plan()), *calls()], [("guidance", "Répondre uniquement aux parties explicites de la demande.")],
                                 generated=[bad, ANSWER])
    assert await runner.ask(GOAL) == ANSWER
    assert snapshots[0]["reason"] == "unjustified_abstention"
    assert snapshots[0]["grounding"] == "not_used"
    assert snapshots[0]["candidate"] == bad
    attempts = runner.last_execution["final_attempts"]
    assert attempts[0]["sources"] == attempts[1]["sources"]
    assert len(runner.grounding_llm.calls) == 1
    assert runner.execution.tool_calls == 2 and len(llm.generate_calls) == 2
    assert runner.last_execution["human"]["assisted_success"]


@pytest.mark.anyio
@pytest.mark.parametrize("first", [None, "", "Je ne peux pas le déterminer à partir des sources disponibles."])
async def test_three_stdio_observations_finalize_without_repeating_collection(project, first):
    (project / "test_target.py").write_text(
        "from target import marker\n\ndef test_marker():\n    assert marker() == 42\n", encoding="utf-8")
    objective = "Lis target.py et test_target.py"
    goal = objective + ", puis exécute les tests de test_target.py."
    proposal = {"goal": goal, "steps": [
        {"id": "1", "objective": objective, "depends_on": [],
         "success": {"tool": "read_file", "kind": "nonempty", "path": "target.py"}},
        {"id": "2", "objective": objective, "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "path": "test_target.py"}},
        {"id": "3", "objective": "exécute les tests de test_target.py", "depends_on": ["2"],
         "success": {"tool": "run_tests", "kind": "observed"}},
    ]}
    candidate = "target.py a été lu [S1]. test_target.py a été lu [S2]. Pytest : 1 passed, returncode=0 [S3]."
    runner, llm, snapshots = agent([
        message(proposal), call("read_file", path="target.py"), call("read_file", path="test_target.py"),
        call("run_tests", path="test_target.py")], [("retry", "")],
        generated=[candidate] if first is None else [first, candidate])
    runner.server = build_server_parameters()
    assert await runner.ask(goal) == candidate
    assert runner.execution.tool_calls == 3
    assert len(runner._active_sources) == 3
    assert len(llm.generate_calls) == (1 if first is None else 2)
    assert len(snapshots) == (0 if first is None else 1)
    assert runner.last_execution["stop_reason"] == "goal_achieved"
    assert len(runner.grounding_llm.calls) == 3
    if first is not None:
        attempts = runner.last_execution["final_attempts"]
        assert attempts[0]["sources"] == attempts[1]["sources"] == ["S1", "S2", "S3"]
        assert attempts[0]["reason"] == ("empty_candidate" if first == "" else "unjustified_abstention")
    first_prompt = llm.generate_calls[0]["prompt"]
    assert "Après récupération" not in first_prompt
    assert "Une plage initiale invalide" not in first_prompt


@pytest.mark.anyio
@pytest.mark.parametrize("candidate,reason", [
    ("marker retourne 42.", "missing_citations"),
    ("Je ne peux pas le déterminer à partir des sources disponibles.", "unjustified_abstention"),
])
async def test_final_diagnostics_distinguish_missing_citations_and_abstention(project, candidate, reason):
    runner, llm, snapshots = agent([
        message(plan()), call("search_code", query="marker", path="absent.py"), calls()[1],
    ], [direct_tool("1", "search_code", query="marker", path="."), ("stop", "")], generated=[candidate])
    result = await runner.ask(GOAL)
    assert snapshots[1]["reason"] == reason
    assert runner.last_execution["stop_reason"] == "human_stopped"
    assert runner.execution.tool_calls == 3
    assert len(llm.generate_calls) == 1
    assert not runner.grounding_llm.calls
    assert not runner.last_execution["human"]["assisted_success"]
    assert "Objectif final non validé" in result


@pytest.mark.anyio
async def test_multiple_patches_form_one_validated_transaction_before_real_mcp(project):
    proposal = plan()
    proposal["steps"][0]["objective"] = "Localise un autre symbole"
    proposal["steps"][1]["success"]["from_step"] = "missing"
    corrections = [dict(step_id="1", changes=dict(objective="Localise marker")),
                   dict(step_id="2", changes=dict(success=plan()["steps"][1]["success"]))]
    runner, llm, snapshots = agent([message(proposal), calls()[1]], [
        ("correct_step", dict(corrections=corrections,
            tool_action=dict(step_id="1", name="search_code", arguments=dict(query="marker", path="."))))], max_replans=0)
    runner.server = build_server_parameters()
    assert await runner.ask(GOAL) == ANSWER
    assert snapshots[0]["plan"] == [] and snapshots[0]["evidence"] == []
    events = runner.last_execution["events"]
    validated = next(e for e in events if e["event"] == "human_plan_validated")
    assert all(step["status"] == "PENDING" and step["evidence"] == [] for step in validated["plan"])
    assert len(snapshots) == 1 and runner.execution.tool_calls == 2
    assert runner.last_execution["completed"] == ["1", "2"]
    assert runner.last_execution["human"]["assisted_success"]


def test_multiple_patches_roll_back_all_changes_if_one_step_remains_invalid():
    proposal = plan()
    before = json.dumps(proposal, ensure_ascii=False)
    with pytest.raises(ValueError, match="Step 2"):
        correct_remaining_plan(proposal, [
            dict(step_id="1", changes=dict(success=dict(tool="search_code", kind="nonempty"))),
            dict(step_id="2", changes=dict(depends_on=["missing"]))],
            GOAL, ["search_code", "read_file"], 6, [])
    assert json.dumps(proposal, ensure_ascii=False) == before


@pytest.mark.anyio
@pytest.mark.parametrize("passes,false_report", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("resume", [False, True])
async def test_read_then_real_pytest_stdio_preserves_exit_status_and_output(project, passes, false_report, resume):
    path = "test_observed.py"
    (project / path).write_text(f"def test_observed():\n    assert {passes}\n", encoding="utf-8")
    goal = f"Lis target.py, puis Exécute les tests de {path} avec run_tests."
    proposal = dict(goal=goal, steps=[
        dict(id="1", objective="Lis target.py", depends_on=[], success=dict(tool="read_file", kind="nonempty")),
        dict(id="2", objective=f"Exécute les tests de {path} avec run_tests", depends_on=["1"],
             success=dict(tool="run_tests", kind="observed"))])
    summary = "1 passed" if passes else "1 failed"
    candidate = f"target.py a été lu [S1].\n\nPytest : {'1 passed' if false_report else summary}, returncode = {0 if passes or false_report else 1} [S2]."
    responses = [message(proposal), call("read_file", path="target.py")]
    responses.append(message(dict(stop="user_decision", reason="Choisir le lancement")) if resume
                     else call("run_tests", path=path))
    decisions = [direct_tool("2", "run_tests", path=path)] if resume else []
    if false_report:
        decisions.append(("stop", ""))
    runner, llm, snapshots = agent(responses, decisions, generated=[candidate])
    runner.server = build_server_parameters()
    answer = await runner.ask(goal)
    if false_report:
        assert runner.last_execution["stop_reason"] != "goal_achieved"
        assert not runner.last_execution["human"]["assisted_success"]
        assert "Objectif final non validé" in answer
    else:
        assert answer == candidate
    result = result_data(runner._active_sources[1]["content"].partition("Result:\n")[2])
    assert result["passed"] is passes
    assert result["returncode"] == (0 if passes else 1)
    assert result["timed_out"] is False and summary in result["stdout"]
    assert len(snapshots) == int(false_report) + int(resume)
    assert runner.last_execution["completed"] == ["1", "2"]
    assert [trace["step_id"] for trace in runner.last_trace] == ["1", "2"]
    if resume:
        assert snapshots[0]["completed"] == ["1"]
        assert [source["observation_id"] for source in runner._active_sources] == ["S1", "S2"]
        assert runner.last_execution["human"]["assisted_success"] is (not false_report)


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["read_file", "run_tests"])
async def test_wrong_explicit_file_never_executes_or_completes(project, tool):
    (project / "test_observed.py").write_text("def test_observed():\n    assert True\n", encoding="utf-8")
    goal = "Lis target.py, puis Exécute les tests de test_observed.py avec run_tests."
    proposal = dict(goal=goal, steps=[
        dict(id="1", objective="Lis target.py", depends_on=[], success=dict(tool="read_file", kind="nonempty")),
        dict(id="2", objective="Exécute les tests de test_observed.py avec run_tests", depends_on=["1"],
             success=dict(tool="run_tests", kind="observed"))])
    responses = [message(proposal)]
    if tool == "run_tests":
        responses.append(call("read_file", path="target.py"))
    responses.append(call(tool, path="other.py"))
    runner, llm, snapshots = agent(responses, [("stop", "")])
    result = await runner.ask(goal)
    assert runner.execution.tool_calls == (1 if tool == "run_tests" else 0)
    assert runner.last_execution["completed"] == (["1"] if tool == "run_tests" else [])
    assert "Objectif final non validé" in result and not llm.generate_calls


def test_wrong_test_file_observation_cannot_satisfy_current_step():
    runner = ToolLoop(llm=FakeLLM([]), grounding_llm=FakeGroundingLLM())
    step = dict(id="1", objective="Exécute tests/test_expected.py", success=dict(tool="run_tests", kind="observed"))
    runner.last_execution = {"plan": [step]}
    from app.agent.tool_loop import build_tool_source
    source = build_tool_source("run_tests", dict(path="tests/test_wrong.py"), json.dumps(
        dict(passed=True, timed_out=False, returncode=0, stdout="1 passed")))
    assert not runner._verifies_step(step, source, [source])


@pytest.mark.anyio
async def test_human_run_tests_omitted_path_keeps_published_tests_default(project):
    (project / "tests").mkdir()
    (project / "tests/test_default.py").write_text("def test_default():\n    assert True\n", encoding="utf-8")
    (project / "test_outside.py").write_text("def test_outside():\n    assert False\n", encoding="utf-8")
    goal = "Localise marker, puis Exécute les tests."
    proposal = dict(goal=goal, steps=[
        dict(id="1", objective="Localise marker", depends_on=[],
             success=dict(tool="search_code", kind="contains", value="marker")),
        dict(id="2", objective="Exécute les tests", depends_on=["1"],
             success=dict(tool="run_tests", kind="observed"))])
    candidate = "marker est dans target.py [S1].\n\nPytest : 1 passed, returncode = 0 [S2]."
    runner, llm, snapshots = agent([message(proposal), calls()[0],
        message(dict(stop="user_decision", reason="Choisir le lancement"))],
        [direct_tool("2", "run_tests"), ("stop", "")], generated=[candidate])
    assert await runner.ask(goal) == candidate
    result = result_data(runner._active_sources[-1]["content"].partition("Result:\n")[2])
    assert result["passed"] is True and "1 passed" in result["stdout"]
    assert "test_outside" not in result["stdout"]
    assert runner.last_trace[-1]["arguments"]["path"] == "tests"
    assert runner.last_execution["human"]["assisted_success"]
