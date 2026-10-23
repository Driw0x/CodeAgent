"""Compare M6 and M8 on one temporary project; fixtures never measure model quality."""

import argparse
import asyncio
from collections import Counter
import copy
import json
import logging
import math
from pathlib import Path
import re
import tempfile
import statistics
from time import perf_counter

from mcp.server import MCPServer

from app.agent.tool_loop import DEFAULT_GROUNDING_MODEL, DEFAULT_TOOL_MODEL, ToolLoop, tool_call_key
from app.llm import LocalLLM
from app.memory.context import ContextBudget
from app.memory.history import describe_error
from app.mcp_server import mcp, tool_with_diagnostics
from app.rag.citations import extract_citations, is_abstention
from app.tools import project_tools


ROOT = Path(__file__).resolve().parents[1]
ABSTENTION = "Je ne peux pas le déterminer à partir des sources disponibles."


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}
    ]}


def step(identifier, objective, tool, value=None, dependency=None):
    success = {"tool": tool, "kind": "contains" if value else "nonempty"}
    if value:
        success["value"] = value
    if dependency and tool == "read_file":
        success["from_step"] = dependency
    return {"id": identifier, "objective": objective,
            "depends_on": [dependency] if dependency else [], "success": success}


def scenarios():
    """Same scripted tool sequence in both modes; M8 alone receives a plan."""
    locate = call("search_code", query="def marker")
    read = call("read_file", path="target.py")
    objectives = ["Localise marker", "lis son implémentation"]
    plan = [step("1", objectives[0], "search_code", "marker"),
            step("2", objectives[1], "read_file", dependency="1")]
    base = {"question": ", puis ".join(objectives) + ".", "needs_plan": True,
            "plan": plan, "calls": [locate, read], "required_reads": ["target.py"],
            "expected_success": True, "options": {}}

    def case(identifier, **updates):
        return {**copy.deepcopy(base), "id": identifier, **updates}

    cases = [
        case("simple", question="Lis target.py.", needs_plan=False, calls=[read]),
        case("long_simple", needs_plan=False, calls=[read], question=(
            "Lis uniquement target.py pour donner la valeur retournée par marker. "
            "Je souhaite une réponse précise, courte, avec citation du fichier et de ses lignes. "
            "Cette demande porte sur une seule inspection du code actuel.")),
        case("dependent"),
    ]
    multi_plan = [*plan, step("3", "localise les tests associés", "search_code", dependency="2"),
                  step("4", "lis ces tests", "read_file", dependency="3")]
    cases.append(case("multisource", question=(
        "Localise marker, puis lis son implémentation, puis localise les tests associés, puis lis ces tests."),
        plan=multi_plan, calls=[locate, read, call("search_code", query="assert marker"),
                               call("read_file", path="test_target.py")],
        required_reads=["target.py", "test_target.py"]))
    cases.append(case("recoverable", calls=[locate, call("read_file", path="target.py", start_line=0), read]))
    replan_objective = "recherche obsolete_marker ou sa définition actuelle"
    initial = [plan[0], step("2", replan_objective, "search_code", "obsolete_marker", "1"),
               step("3", "lis son implémentation", "read_file", dependency="2")]
    corrected = copy.deepcopy(initial)
    corrected[1]["success"]["value"] = "marker"
    cases.append(case("replan", question=(
        "Localise marker, puis recherche obsolete_marker ou sa définition actuelle, puis lis son implémentation."),
        plan=initial, revised_plan=corrected,
        calls=[locate, call("search_code", query="obsolete_marker"),
               call("search_code", query="def marker", path="target.py"), read]))
    missing_plan = [step("1", "Localise absent_marker", "search_code", "absent_marker"),
                    step("2", "lis son implémentation", "read_file", "absent_marker", "1")]
    cases.append(case("irreparable", question="Localise absent_marker, puis lis son implémentation.",
                      plan=missing_plan, calls=[call("search_code", query="absent_marker")],
                      expected_success=False))
    for identifier, options in (
        ("tool_budget", {"max_tool_calls": 1}),
        ("token_budget", {"max_total_tokens": 1}),
        ("step_budget", {"max_steps": 1}),
        ("time_budget", {"max_seconds": 0.000001}),
    ):
        cases.append(case(identifier, options=options, expected_success=False))
    invalid = copy.deepcopy(plan)
    invalid[0]["depends_on"] = ["2"]
    cases.append(case("invalid_plan", plan=invalid, invalid_plan=True))
    cases.append(case("timeout", options={"tool_timeout": 0.01, "max_retries": 0},
                      timeout=True, expected_success=False))
    cases.append(case("user_decision", question="Localise marker, puis lis le fichier que je choisirai.", expected_success=False,
                      planner_stop={"stop": "user_decision", "reason": "Le fichier doit être choisi par l'utilisateur."},
                      calls=[]))
    limited = copy.deepcopy(next(item for item in cases if item["id"] == "replan"))
    limited.update(id="replan_limit", options={"max_replans": 0}, expected_success={"off": True, "auto": False})
    cases.append(limited)
    duplicate_plan = [plan[0], step("2", "cherche marker", "search_code", "marker", "1")]
    cases.append(case("duplicate", question="Localise marker, puis cherche marker.", plan=duplicate_plan,
                      calls=[locate, locate], expected_success=False))
    return cases


def source_sections(prompt):
    return [(int(match.group(1)), match.group(2)) for match in re.finditer(
        r"\[S(\d+)\]\n(File:.*?)(?=\n\n---\n\n|\n\nBILAN DES APPELS|\n\nAFFIRMATION À VÉRIFIER:|\Z)",
        prompt, re.DOTALL)]


class FixtureLLM:
    """Predefined choices, literal final claims, and synthetic byte/4 token estimates."""

    context_tokens = 16384
    num_predict = 512
    model = "scripted-fixture-not-a-model"

    def __init__(self, case):
        self.case = case
        self.cursor = 0
        self.planning_calls = 0
        self.sizes = []
        self.last_stats = None
        self.reset_usage_stats()

    def reset_usage_stats(self):
        self.usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def record(self, messages, response, tools=None):
        size = ContextBudget.for_llm(self).check(messages, tools)
        self.sizes.append(size)
        self.last_stats = {"prompt_tokens": math.ceil(size / 4),
                           "completion_tokens": math.ceil(len(json.dumps(response, ensure_ascii=False).encode()) / 4)}
        self.last_stats["total_tokens"] = sum(self.last_stats.values())
        for key, value in self.last_stats.items():
            self.usage_stats[key] += value
        return response

    def chat(self, messages, tools=None):
        if tools is None:
            self.planning_calls += 1
            plan = self.case.get("revised_plan", self.case["plan"]) if self.planning_calls > 1 else self.case["plan"]
            response = {"role": "assistant", "content": json.dumps(
                self.case.get("planner_stop", {"goal": self.case["question"], "steps": plan}), ensure_ascii=False)}
        else:
            calls = self.case["calls"]
            response = copy.deepcopy(calls[self.cursor]) if self.cursor < len(calls) else {
                "role": "assistant", "content": "Collecte arrêtée ; seules les observations font preuve."}
            self.cursor += 1
        return self.record(messages, response, tools)

    def generate(self, prompt, system_prompt=None):
        if "AFFIRMATION À VÉRIFIER:" in prompt:
            sources, claim = prompt.split("AFFIRMATION À VÉRIFIER:", 1)
            literal = (("marker retourne 42" in claim and "def marker" in sources and "return 42" in sources)
                       or ("Le test compare marker() à 42" in claim and "assert marker() == 42" in sources))
            response = "SUPPORTED" if literal else "UNSUPPORTED"
        else:
            claims = []
            for identifier, source in source_sections(prompt):
                if "Type: tool_error" in source or "Name: read_file" not in source:
                    continue
                if "def marker" in source and "return 42" in source:
                    claims.append(f"marker retourne 42 [S{identifier}].")
                if "assert marker() == 42" in source:
                    claims.append(f"Le test compare marker() à 42 [S{identifier}].")
            response = "\n\n".join(claims) or ABSTENTION
        return self.record([{"role": "system", "content": system_prompt or ""},
                            {"role": "user", "content": prompt}], response)


class ObservedLocalLLM(LocalLLM):
    """Keep prompt estimates alongside actual Ollama usage counters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizes = []
        self.planning_calls = 0

    def chat(self, messages, tools=None):
        self.sizes.append(self.context_budget.measure(messages, tools))
        self.planning_calls += tools is None
        return super().chat(messages, tools)

    def generate(self, prompt, system_prompt=None):
        self.sizes.append(self.context_budget.measure([
            {"role": "system", "content": system_prompt or ""}, {"role": "user", "content": prompt}]))
        return super().generate(prompt, system_prompt)


def timeout_server():
    """Real MCP transport, controlled slow read, unchanged project tools otherwise."""
    server = MCPServer("CodeAgent controlled timeout benchmark")

    @server.tool()
    async def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        await asyncio.sleep(0.1)
        return project_tools.read_file(path, start_line, end_line)

    for tool in (project_tools.search_code, project_tools.list_files, project_tools.get_git_diff, project_tools.run_tests):
        server.tool()(tool_with_diagnostics(tool))
    return server


def observed_results(memory):
    results = {}
    for message in memory.recent[-1]["observations"] if memory.recent else []:
        if message.get("role") != "tool":
            continue
        try:
            data = json.loads(message["content"])
            if not data.get("ok"):
                continue
            result = json.loads(data["result"])
            results[data["source"].strip("[]")] = {"tool": data["tool"], "result": result}
        except (ValueError, KeyError, TypeError):
            continue
    return results


def validated_steps(execution, observations):
    """Independent evidence check; a successful MCP call alone is insufficient."""
    checks = []
    completed = set()
    by_id = {item["id"]: item for item in execution.get("plan", [])}
    for item in execution.get("plan", []):
        if item.get("status") != "COMPLETED":
            continue
        condition = item["success"]
        evidence = [observations.get(identifier.strip("[]")) for identifier in item.get("evidence", [])]
        matches = [entry["result"] for entry in evidence if entry and entry["tool"] == condition["tool"]]
        field = {"read_file": "content", "search_code": "matches", "list_files": "files", "get_git_diff": "diff"}.get(condition["tool"])
        payloads = [result.get(field) for result in matches]
        # Independent oracle: inspect decoded business strings, never JSON escaping or metadata.
        texts = [result.get("content", "") if condition["tool"] == "read_file" else
                 "\n".join(match.get("text", "") for match in result.get("matches", []))
                 if condition["tool"] == "search_code" else
                 "\n".join(result.get("files", [])) if condition["tool"] == "list_files" else
                 result.get("diff", result.get("stdout", "")) for result in matches]
        found = any(condition.get("value", "").casefold() in text.casefold() and bool(text) for text in texts)
        if condition["kind"] == "nonempty":
            found = any(payloads)
        if condition["tool"] == "run_tests":
            found = any(result.get("passed") is True and result.get("timed_out") is False for result in matches)
        origin = condition.get("from_step")
        if origin:
            paths = set()
            for identifier in by_id[origin].get("evidence", []):
                previous = observations.get(identifier.strip("[]"), {}).get("result", {})
                paths.update(match["file"] for match in previous.get("matches", []))
                paths.update(previous.get("files", []))
                if previous.get("path"):
                    paths.add(previous["path"])
            found = found and all(result.get("path") in paths for result in matches)
        valid = bool(found and all(dependency in completed for dependency in item["depends_on"]))
        checks.append(valid)
        if valid:
            completed.add(item["id"])
    return checks


async def evaluate(case, execution, real=False, debug=False, model=DEFAULT_TOOL_MODEL, grounding_model=DEFAULT_GROUNDING_MODEL,
                   hitl="off", intervention_handler=None):
    llm = ObservedLocalLLM(model=model) if real else FixtureLLM(case)
    grounder = ObservedLocalLLM(model=grounding_model) if real else FixtureLLM(case)
    options = {"max_steps": 12, "max_plan_steps": 6, "max_replans": 2,
               "max_tool_calls": 16, "max_total_tokens": 32768, "max_seconds": 900,
               **case["options"]}
    agent = ToolLoop(llm=llm, grounding_llm=grounder, server=timeout_server() if case.get("timeout") else mcp,
                     planning=execution, hitl=hitl, intervention_handler=intervention_handler, **options)
    agent.debug = debug
    start = perf_counter()
    answer = ""
    error = None
    try:
        answer = await agent.ask(case["question"])
    except Exception as failure:
        error = describe_error(failure)
    seconds = perf_counter() - start
    state = agent.last_execution
    observations = observed_results(agent.memory)
    reads = {entry["result"].get("path"): entry["result"].get("content", "")
             for entry in observations.values() if entry["tool"] == "read_file"}
    evidence_complete = all(path in reads for path in case["required_reads"])
    if "answer_contains" in case:
        evidence_complete = evidence_complete and all(
            all(fragment in reads.get(path, "") for fragment in fragments)
            for path, fragments in case.get("read_contains", {}).items())
        answer_complete = all(fragment.casefold() in answer.casefold() for fragment in case["answer_contains"])
        answer_complete = answer_complete and not any(fragment.casefold() in answer.casefold()
                                                      for fragment in case.get("answer_excludes", []))
    else:
        answer_complete = ("42" in answer and "marker" in answer
                           and ("test_target.py" not in case["required_reads"] or "test" in answer.casefold()))
    grounding = agent.last_grounding == "passed" and bool(extract_citations(answer)) and not is_abstention(answer)
    planned = state.get("mode") in {"M8_PLANNED", "M8_ADAPTIVE"}
    plan_complete = not planned or (bool(state.get("plan")) and not state.get("remaining")
                                    and len(state.get("completed", [])) == len(state["plan"])
                                    and state.get("stop_reason") == "goal_achieved")
    success = bool(error is None and evidence_complete and answer_complete and grounding and plan_complete)
    expected = case["expected_success"]
    if isinstance(expected, dict):
        expected = expected.get(execution, expected["auto"])
    expected = expected and not (case.get("invalid_plan") and execution == "auto")
    step_checks = validated_steps(state, observations)
    trace = agent.last_trace
    if case.get("required_first_call"):
        expected_call = case["required_first_call"]
        evidence_complete = evidence_complete and bool(trace) and trace[0]["tool"] == expected_call["tool"] and all(
            trace[0]["arguments"].get(key) == value for key, value in expected_call.get("arguments", {}).items())
    if case.get("first_read_arguments"):
        first_read = next((item for item in trace if item["tool"] == "read_file"), {})
        evidence_complete = evidence_complete and all(first_read.get("arguments", {}).get(key) == value
                                                      for key, value in case["first_read_arguments"].items())
    if case.get("require_error_recovery"):
        evidence_complete = evidence_complete and any(item["status"] in {"error", "exception", "timeout"} for item in trace)
    success = bool(success and evidence_complete)
    executed = [item for item in trace if item["status"] in {"success", "error", "exception", "timeout"}]
    counts = Counter(tool_call_key(item["tool"], item["arguments"]) for item in trace)
    sizes = llm.sizes + grounder.sizes
    started = {}
    durations = {}
    for event in state.get("events", []):
        identifier = event.get("step_id")
        if event["event"] == "step_started":
            started[identifier] = event["seconds"]
        elif event["event"] in {"step_completed", "step_failed"} and identifier in started:
            durations.setdefault(identifier, []).append(round(event["seconds"] - started.pop(identifier), 4))
    stops = {"irreparable": "insufficient_evidence", "tool_budget": "global_budget",
             "token_budget": "global_budget", "time_budget": "global_budget", "step_budget": "step_limit",
             "invalid_plan": "invalid_plan", "timeout": "insufficient_evidence",
             "user_decision": "user_decision", "replan_limit": "replan_limit", "duplicate": "repeated_errors"}
    control_checks = {}
    if not real and execution == "auto" and case["id"] in stops:
        control_checks["expected_stop"] = state.get("stop_reason") == stops[case["id"]]
    if case["id"] == "recoverable" and not real:
        control_checks["error_then_success"] = any(item["status"] == "error" for item in trace) and success
    if case["id"] == "replan" and execution == "auto" and not real:
        control_checks["completed_step_preserved"] = any(
            event["event"] == "replanning" and event["preserved"] == ["1"] for event in state["events"])
        control_checks["no_repeated_call"] = max(counts.values(), default=0) == 1
    row = {
        "case": case["id"], "execution": execution, "mode": state.get("mode"),
        "question": case["question"], "simulated": not real, "success": success,
        "expected_success": expected, "expectation_met": success == expected and error is None,
        "evidence_complete": evidence_complete, "answer_complete": bool(answer_complete),
        "planning_needed": case["needs_plan"], "planning_selected": planned,
        "planning_false_positive": planned and not case["needs_plan"] if execution == "auto" else None,
        "planning_false_negative": not planned and case["needs_plan"] if execution == "auto" else None,
        "planning_calls": llm.planning_calls, "completed_steps": len(state.get("completed", [])),
        "planned_steps": len(state.get("plan", [])), "independently_valid_steps": sum(step_checks),
        "completed_step_evidence_valid": all(step_checks), "replans": state.get("replans", 0),
        "local_revisions": state.get("local_revisions", 0),
        "replanning_success": success and state.get("replans", 0) > 0 if case["id"] == "replan" else None,
        "repeated_calls": sum(count - 1 for count in counts.values()),
        "tool_calls": len(executed), "tool_attempts": len(trace),
        "tool_successes": sum(item["status"] == "success" for item in trace),
        "tool_failures": sum(item["status"] in {"error", "exception", "timeout"} for item in trace),
        "tool_schema_or_validation_errors": sum(item["status"] == "error" for item in trace),
        "max_context_estimate": max(sizes, default=0), "input_limit": agent.budget.input_limit,
        "context_bounded": all(size <= agent.budget.input_limit for size in sizes),
        "grounding": agent.last_grounding, "grounded_non_abstention": bool(grounding),
        "seconds": seconds, "stage_seconds": agent.last_stage_times,
        "step_seconds": durations or None,
        "actual_model_tokens": agent.last_token_stats if real and agent.last_token_stats["total"] else None,
        "fixture_token_estimates": None if real else agent.last_token_stats,
        "stage_tokens": agent.last_stage_stats, "answer": answer, "error": error,
        "stop_reason": state.get("stop_reason"), "execution_state": state,
        "control_checks": control_checks,
        "trace": trace, "observations": observations,
        "source_mapping": {f"S{index}": source.get("observation_id")
                           for index, source in enumerate(agent._final_sources, 1)},
        "selected_sources": agent._final_sources,
        "grounding_claims": agent.last_debug.get("claims", []),
        "final_candidate": agent.last_debug.get("candidate"),
        "validation_diagnostic": agent.last_debug.get("reason"),
        "independent_semantic_review": None,
    }
    if hitl != "off":
        row["human"] = state.get("human")
    if debug:
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    return row


async def run(case_id=None, execution="both", real=False, debug=False, model=DEFAULT_TOOL_MODEL,
              grounding_model=DEFAULT_GROUNDING_MODEL, cases_file=None, hitl="off"):
    suite = json.loads(Path(cases_file).read_text(encoding="utf-8")) if cases_file else None
    cases = [case for case in (suite["cases"] if suite else scenarios()) if case_id is None or case["id"] == case_id]
    if not cases:
        raise ValueError(f"Unknown case: {case_id}")
    modes = ("off", "auto") if execution == "both" else ("auto", "adaptive") if execution == "compare" else (execution,)
    original_root = project_tools.PROJECT_ROOT
    logger = logging.getLogger("mcp.server.mcpserver.server")
    previous_level = logger.level
    rows = []
    with tempfile.TemporaryDirectory(prefix="codeagent-planning-benchmark-") as temporary:
        project_tools.PROJECT_ROOT = Path(temporary)
        (Path(temporary) / "target.py").write_text("def marker():\n    return 42\n", encoding="utf-8")
        (Path(temporary) / "test_target.py").write_text(
            "from target import marker\n\ndef test_marker():\n    assert marker() == 42\n", encoding="utf-8")
        for path, content in (suite or {}).get("files", {}).items():
            target = (Path(temporary) / path).resolve()
            if not target.is_relative_to(Path(temporary).resolve()):
                raise ValueError("Fixture path outside temporary project")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        try:
            logger.setLevel(logging.CRITICAL)
            # Always sequential: equal fixtures, fresh memory/model objects, no concurrent Ollama requests.
            for case in cases:
                for mode in modes:
                    rows.append(await evaluate(case, mode, real, debug, model, grounding_model, hitl))
        finally:
            project_tools.PROJECT_ROOT = original_root
            logger.setLevel(previous_level)
    auto = [row for row in rows if row["execution"] == "auto"]
    checks = {
        "context_bounded": all(row["context_bounded"] for row in rows),
        "completed_steps_have_evidence": all(row["completed_step_evidence_valid"] for row in rows),
        "routing_no_false_positives": not any(row["planning_false_positive"] for row in auto),
        "routing_no_false_negatives": not any(row["planning_false_negative"] for row in auto),
        "scenario_expectations_met": all(row["expectation_met"] for row in rows),
        "controlled_stops_and_recovery": all(all(row["control_checks"].values()) for row in rows),
    }
    summary = {}
    for mode in modes:
        measured = [row for row in rows if row["execution"] == mode]
        planned_count = sum(row["planned_steps"] for row in measured)
        calls_count = sum(row["tool_calls"] for row in measured)
        comparable_ids = {"simple", "long_simple", "dependent", "multisource", "recoverable", "replan"}
        comparable = [row for row in measured if row["case"] in comparable_ids]
        summary[mode] = {
            "trials": len(measured), "successes": sum(row["success"] for row in measured),
            "end_to_end_success_rate": sum(row["success"] for row in measured) / len(measured),
            "common_completable_trials": len(comparable),
            "common_completable_successes": sum(row["success"] for row in comparable),
            "completed_steps": sum(row["completed_steps"] for row in measured),
            "planned_steps": planned_count,
            "step_success_rate": sum(row["independently_valid_steps"] for row in measured) / planned_count if planned_count else None,
            "tool_calls": calls_count,
            "successful_tool_rate": sum(row["tool_successes"] for row in measured) / calls_count if calls_count else None,
            "repeated_calls": sum(row["repeated_calls"] for row in measured),
            "p50_seconds": statistics.median(row["seconds"] for row in measured),
            "total_actual_model_tokens": (sum(row["actual_model_tokens"]["total"] for row in measured)
                                         if real and all(row["actual_model_tokens"] is not None for row in measured) else None),
            "total_fixture_token_estimates": None if real else sum(row["fixture_token_estimates"]["total"] for row in measured),
        }
    return {
        "protocol": "planning-v1-real" if real else "planning-v1-deterministic",
        "models": [model, grounding_model] if real else None, "simulated": not real,
        "cases_file": str(cases_file) if cases_file else None,
        "criteria": "Current MCP read evidence, required answer facts, accepted citations/grounding, and every planned step complete.",
        "limitations": [
            "Fixture choices and grounding are scripted; they measure execution mechanics, never model intelligence or production latency.",
            "Fixture tokens are synthetic UTF-8 byte/4 estimates; actual_model_tokens is null without Ollama.",
            "M6/M8 use identical fixture call sequences, budgets, source data and fresh memory; no measured advantage is presumed.",
            "Real mode uses the real router/planner/tools/final/grounder. Scripted invalid-plan/error/replan injections apply only to fixture mode.",
            "Real expected failures are not guaranteed: Ollama may choose different valid calls; review answers and raw evidence.",
            "Per-step latency comes from recorded start/completion/failure events; unfinished attempts have no completion latency.",
            "Known M7 result remains 7/7 read_file successes and 0/7 fully autonomous finals; this benchmark does not revise it.",
        ], "checks": checks, "summary": summary, "results": rows,
        "routing": {"false_positives": sum(bool(row["planning_false_positive"]) for row in auto),
                    "false_negatives": sum(bool(row["planning_false_negative"]) for row in auto), "samples": len(auto)},
    }


def print_comparison(payload):
    print("case             | M6: success steps tools replans tokens seconds grounding | M8: success steps tools replans tokens seconds grounding")
    grouped = {}
    for row in payload["results"]:
        tokens = row["actual_model_tokens"] or row["fixture_token_estimates"]
        grouped.setdefault(row["case"], {})[row["execution"]] = (
            f"{int(row['success'])} {row['completed_steps']}/{row['planned_steps']} {row['tool_calls']} "
            f"{row['replans']} {tokens['total'] if tokens else 'unavailable'} {row['seconds']:.3f} {row['grounding']}")
    for identifier, modes in grouped.items():
        print(f"{identifier:16} | {modes.get('off', 'not run'):55} | {modes.get('auto', 'not run')}")
        if "adaptive" in modes:
            print(f"  adaptive       | {modes['adaptive']}")
    print("Tokens: " + ("synthetic fixture estimates; no Ollama execution" if payload["simulated"] else "Ollama counters"))
    print("Checks: " + json.dumps(payload["checks"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Case ID from historical scenarios or --cases-file.")
    parser.add_argument("--cases-file", type=Path, help="Additional predefined questions, literal checks and read-only fixture files; historical cases unchanged.")
    parser.add_argument("--debug", action="store_true", help="Print measured execution state, tool trace and grounding claims.")
    parser.add_argument("--real", action="store_true", help="Use existing Ollama service, sequentially; never starts Ollama.")
    parser.add_argument("--hitl", choices=("off", "on-failure"), default="off", help="Offer the M8 terminal intervention on failures; historical defaults unchanged.")
    parser.add_argument("--execution", choices=("auto", "off", "both", "adaptive", "compare"), default="both")
    parser.add_argument("--model", default=DEFAULT_TOOL_MODEL)
    parser.add_argument("--grounding-model", default=DEFAULT_GROUNDING_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = asyncio.run(run(args.case, args.execution, args.real, args.debug, args.model, args.grounding_model, args.cases_file, args.hitl))
    output = args.output or ROOT / "benchmarks/planning/results" / ("real.json" if args.real else "deterministic.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_comparison(payload)
    print(f"Report: {output.resolve()}")
    if not all(payload["checks"].values()) and not args.real:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
