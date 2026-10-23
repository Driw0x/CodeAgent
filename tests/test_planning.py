import copy
import json
import unicodedata

import pytest

from app.agent.planning import (
    ACTION_PATTERN,
    _collection_actions,
    ExecutionStopped,
    PlanningExecution,
    evidence_paths,
    planning_decision,
    validate_plan,
    verifies,
)
from app.agent.tool_loop import build_tool_source

import ast

from app.memory.context import reduce_observation
from app.tools import project_tools


GOAL = "Localise ORION, puis lis son implémentation."
TOOLS = ["search_code", "read_file", "list_files", "run_tests"]


def evidence_plan(goal=GOAL):
    return {
        "goal": goal,
        "steps": [
            {"id": "1", "objective": "Localise ORION", "depends_on": [],
             "success": {"tool": "search_code", "kind": "contains", "value": "ORION"}},
            {"id": "2", "objective": "lis son implémentation", "depends_on": ["1"],
             "success": {"tool": "read_file", "kind": "contains", "value": "ORION", "from_step": "1"}},
        ],
    }


@pytest.mark.parametrize("question", [
    "Quelle est la valeur de ORION ?",
    "Lis app/memory/context.py.",
    "Lis app/memory/context.py, puis explique le résultat avec ses sources.",
    "Recherche ORION dans le projet.",
    "Rappelle ce que j'ai indiqué dans cette session puis explique ma décision.",
    "Quel est le rôle de FAISS ? " + "Je souhaite une réponse détaillée et sourcée. " * 30,
])
def test_simple_requests_do_not_trigger_planner(question):
    assert planning_decision(question)[0] == "M6_DIRECT"


@pytest.mark.parametrize("question", [
    GOAL,
    "Localise le composant responsable des embeddings, puis explique comment il interagit avec FAISS.",
    "Identifie le mécanisme de gestion des erreurs MCP, lis son implémentation et vérifie les tests associés.",
    "Find ORION then read its implementation.",
    "Recherche ORION puis lis son implémentation.",
    "Lis app/memory/context.py, puis explique comment ce composant interagit avec FAISS.",
    "Lis app/memory/context.py, puis explique le résultat et vérifie les tests associés.",
])
def test_dependent_requests_trigger_planner(question):
    mode, reason = planning_decision(question)
    assert mode == "M8_PLANNED"
    assert reason
    assert planning_decision(question, "off")[0] == "M6_DIRECT"


@pytest.mark.parametrize("identifier", [
    "compare_manifests", "inspect_state", "analyse_sources", "recherche_tokens", "lecture2", "compare42",
])
def test_identifiers_are_not_collection_actions(identifier):
    first = f"Localise la définition de {identifier} dans app/memory/project_state.py"
    goal = f"{first}, puis lis son implémentation et explique le résultat avec ses sources."
    raw = evidence_plan(goal)
    raw["steps"][0].update(objective=first,
                           success={"tool": "search_code", "kind": "contains", "value": identifier})
    raw["steps"][1]["success"] = {"tool": "read_file", "kind": "nonempty", "from_step": "1"}
    assert len(validate_plan(raw, goal, TOOLS, 6)) == 2
    assert planning_decision(goal)[0] == "M8_PLANNED"
    assert planning_decision(f"Localise {identifier} dans les fichiers associés.")[0] == "M6_DIRECT"


@pytest.mark.parametrize("first", ["Compare ORION et VEGA", "Comparez ORION et VEGA", "Inspecte ORION", "Vérifie ORION"])
def test_ordinary_verbs_remain_collection_actions(first):
    goal = f"{first}, puis lis son implémentation."
    raw = evidence_plan(goal)
    raw["steps"][0]["objective"] = first
    assert planning_decision(goal)[0] == "M8_PLANNED"
    assert len(validate_plan(raw, goal, TOOLS, 6)) == 2


def test_ordinary_action_after_identifier_cannot_be_omitted():
    first = "Localise compare_manifests"
    goal = f"{first}, puis lis son implémentation et compare les résultats."
    raw = evidence_plan(goal)
    raw["steps"][0].update(objective=first,
                           success={"tool": "search_code", "kind": "contains", "value": "compare_manifests"})
    raw["steps"][1]["success"] = {"tool": "read_file", "kind": "nonempty", "from_step": "1"}
    with pytest.raises(ValueError, match="omits"):
        validate_plan(raw, goal, TOOLS, 6)
    raw["steps"].append({"id": "3", "objective": "compare les résultats", "depends_on": ["2"],
                         "success": {"tool": "search_code", "kind": "nonempty"}})
    assert len(validate_plan(raw, goal, TOOLS, 6)) == 3


def test_valid_plan_is_bounded_ordered_and_does_not_mutate_model_output():
    raw = evidence_plan()
    before = copy.deepcopy(raw)
    plan = validate_plan(raw, GOAL, TOOLS, 6)
    assert [step["status"] for step in plan] == ["PENDING", "PENDING"]
    assert plan[1]["depends_on"] == ["1"]
    assert raw == before


def test_ambiguous_excerpt_cannot_cover_multiple_requested_reads():
    goal = "Localise ORION, puis lis a.py et lis b.py."
    raw = evidence_plan(goal)
    raw["steps"][1].update(objective="lis", success={"tool": "read_file", "kind": "nonempty", "from_step": "1"})
    with pytest.raises(ValueError, match="collection action"):
        validate_plan(raw, goal, TOOLS, 6)
    raw["steps"][1]["objective"] = "lis a.py"
    raw["steps"].append({"id": "3", "objective": "lis b.py", "depends_on": ["2"],
                         "success": {"tool": "read_file", "kind": "nonempty"}})
    assert len(validate_plan(raw, goal, TOOLS, 6)) == 3


def test_read_without_origin_requires_path_in_its_own_objective():
    goal = "Dans a.py, lis la signature, puis inspecte le résultat."
    raw = {"goal": goal, "steps": [
        {"id": "1", "objective": "lis la signature", "depends_on": [],
         "success": {"tool": "read_file", "kind": "nonempty"}},
        {"id": "2", "objective": "inspecte le résultat", "depends_on": ["1"],
         "success": {"tool": "search_code", "kind": "nonempty"}},
    ]}
    with pytest.raises(ValueError, match="read path"):
        validate_plan(raw, goal, TOOLS, 6)


@pytest.mark.parametrize("goal,objective,error", [
    ("Lis a.py, puis lis b.py.", "lis", "unambiguous collection action"),
    ("Lis a.py, puis lis b.py et c.py.", "lis b.py et c.py", "single explicit path"),
])
def test_read_objective_cannot_omit_or_combine_explicit_file_targets(goal, objective, error):
    raw = {"goal": goal, "steps": [
        {"id": "1", "objective": "Lis a.py", "depends_on": [],
         "success": {"tool": "read_file", "kind": "nonempty"}},
        {"id": "2", "objective": objective, "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "1"}},
    ]}
    with pytest.raises(ValueError, match=error):
        validate_plan(raw, goal, TOOLS, 6)


def test_result_explanation_uses_collected_evidence_and_the_separate_final():
    goal = "Localise ORION, puis lis son implémentation et explique le résultat avec ses sources."
    raw = evidence_plan(goal)
    plan = validate_plan(raw, goal, TOOLS, 6)
    assert [step["objective"] for step in plan] == ["Localise ORION", "lis son implémentation"]
    redundant = copy.deepcopy(raw)
    redundant["steps"].append({"id": "3", "objective": "explique le résultat avec ses sources",
                               "depends_on": ["2"], "success": {"tool": "search_code", "kind": "nonempty"}})
    with pytest.raises(ValueError, match="collection action"):
        validate_plan(redundant, goal, TOOLS, 6)


@pytest.mark.parametrize("explanation", [
    "explique comment ORION interagit avec FAISS",
    "explique le résultat de ORION avec ses sources",
    "explique le résultat et vérifie les tests associés",
])
def test_explanations_with_additional_evidence_targets_cannot_be_omitted(explanation):
    goal = f"Localise ORION, puis lis son implémentation et {explanation}."
    with pytest.raises(ValueError, match="omits"):
        validate_plan(evidence_plan(goal), goal, TOOLS, 6)


def test_recorded_recovery_plan_keeps_tool_arguments_out_of_success_conditions():
    goal = ("Localise la définition de read_dir dans app/parser/file_loader.py, puis lis son implémentation "
            "en commençant par la ligne 999 ; si cette plage est invalide, corrige-la.")
    raw = evidence_plan(goal)
    raw["steps"][0].update(objective="Localise la définition de read_dir dans app/parser/file_loader.py",
                           success={"tool": "search_code", "kind": "contains", "value": "read_dir"})
    raw["steps"][1].update(objective="lis son implémentation en commençant par la ligne 999",
                           success={"tool": "read_file", "kind": "nonempty", "from_step": "1", "start_line": 999})
    with pytest.raises(ValueError, match="Invalid success condition"):
        validate_plan(raw, goal, TOOLS, 6)
    del raw["steps"][1]["success"]["start_line"]
    assert len(validate_plan(raw, goal, TOOLS, 6)) == 2
    raw["steps"].append({"id": "3", "objective": "si cette plage est invalide, corrige-la", "depends_on": ["2"],
                         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "1"}})
    with pytest.raises(ValueError, match="explicit action"):
        validate_plan(raw, goal, TOOLS, 6)


def test_recorded_multisource_path_requires_origin_only_when_used_as_condition():
    goal = ("Localise ORION dans current.py, puis lis son implémentation, "
            "puis inspecte les tests associés dans tests/test_tool_loop.py.")
    raw = evidence_plan(goal)
    raw["steps"][0]["objective"] = "Localise ORION dans current.py"
    raw["steps"].append({"id": "3", "objective": "inspecte les tests associés dans tests/test_tool_loop.py",
                         "depends_on": ["2"], "success": {"tool": "read_file", "kind": "nonempty",
                                                            "path": "tests/test_tool_loop.py"}})
    with pytest.raises(ValueError, match="Path selection requires an evidence origin"):
        validate_plan(raw, goal, TOOLS, 6)
    del raw["steps"][2]["success"]["path"]
    assert len(validate_plan(raw, goal, TOOLS, 6)) == 3


@pytest.mark.parametrize("mutation", [
    lambda plan: plan.update(goal="Invented goal"),
    lambda plan: plan.update(steps=[]),
    lambda plan: plan.update(steps=plan["steps"][:1]),
    lambda plan: plan["steps"][1].update(id="1"),
    lambda plan: plan["steps"][1].update(id=2),
    lambda plan: plan["steps"][0].update(depends_on=["2"]),
    lambda plan: plan["steps"][1].update(depends_on=[]),
    lambda plan: plan["steps"][1].update(depends_on=["1", "1"]),
    lambda plan: plan["steps"][1].update(objective="Édite le code"),
    lambda plan: plan["steps"][1].update(extra="invented"),
    lambda plan: plan["steps"][1]["success"].update(tool="write_file"),
    lambda plan: plan["steps"][1]["success"].update(tool="unknown_tool"),
    lambda plan: plan["steps"][1]["success"].update(kind="model_says_done"),
    lambda plan: plan["steps"][1]["success"].update(value="INVENTED_SYMBOL"),
    lambda plan: plan["steps"][1]["success"].update(from_step="2"),
    lambda plan: plan["steps"][1]["success"].pop("from_step"),
    lambda plan: plan["steps"][0]["success"].update(path="unobserved.py"),
])
def test_invalid_plans_are_rejected_before_execution(mutation):
    raw = evidence_plan()
    mutation(raw)
    with pytest.raises(ValueError):
        validate_plan(raw, GOAL, TOOLS, 6)


def test_plan_size_limit_and_unavailable_tool_are_enforced():
    raw = evidence_plan()
    with pytest.raises(ValueError):
        validate_plan(raw, GOAL, TOOLS, 1)
    with pytest.raises(ValueError, match="Unknown tool"):
        validate_plan(raw, GOAL, ["search_code"], 6)


def test_replanning_preserves_completed_goal_condition_and_evidence():
    original = validate_plan(evidence_plan(), GOAL, TOOLS, 6)
    original[0].update(status="COMPLETED", evidence=["S1"])
    replanned = validate_plan(evidence_plan(), GOAL, TOOLS, 6, [original[0]])
    assert replanned[0] == original[0]
    assert replanned[0]["evidence"] is not original[0]["evidence"]
    changed = evidence_plan()
    changed["steps"][0]["success"] = {"tool": "search_code", "kind": "nonempty"}
    with pytest.raises(ValueError, match="completed"):
        validate_plan(changed, GOAL, TOOLS, 6, [original[0]])


def test_completed_step_cannot_disappear_during_replanning():
    original = validate_plan(evidence_plan(), GOAL, TOOLS, 6)
    original[0].update(status="COMPLETED", evidence=["S1"])
    changed = evidence_plan()
    changed["steps"][0]["id"] = "new"
    changed["steps"][1]["depends_on"] = ["new"]
    changed["steps"][1]["success"]["from_step"] = "new"
    with pytest.raises(ValueError, match="completed"):
        validate_plan(changed, GOAL, TOOLS, 6, [original[0]])


@pytest.mark.parametrize("tool,data,expected", [
    ("search_code", {"matches": []}, False),
    ("search_code", {"matches": [{"file": "current.py", "text": "ORION = 42"}]}, True),
    ("read_file", {"path": "ORION.py", "content": ""}, False),
    ("read_file", {"path": "ORION.py", "content": "OTHER = 1"}, False),
    ("read_file", {"content": "ORION = 42"}, True),
    ("list_files", {"files": []}, False),
    ("list_files", {"files": ["ORION.py"]}, True),
    ("get_git_diff", {"diff": ""}, False),
    ("get_git_diff", {"diff": "+ORION = 42"}, True),
    ("run_tests", {"passed": False, "timed_out": False, "stdout": "ORION failed"}, False),
    ("run_tests", {"passed": True, "timed_out": True, "stdout": "ORION passed"}, False),
    ("run_tests", {"passed": True, "timed_out": False, "returncode": 0, "stdout": "ORION passed"}, True),
])
def test_step_verification_checks_business_data_not_transport_success(tool, data, expected):
    assert verifies({"tool": tool, "kind": "contains", "value": "ORION"}, tool, json.dumps(data)) == expected


def test_step_verification_rejects_wrong_tool_and_non_json():
    success = {"tool": "read_file", "kind": "nonempty"}
    assert not verifies(success, "search_code", '{"matches": ["ORION"]}')
    assert not verifies(success, "read_file", "ORION = 42")
    assert not verifies(success, "read_file", "[]")


@pytest.mark.parametrize("kind", ["nonempty", "contains"])
@pytest.mark.parametrize("content,expected", [
    ("def marker():", False),
    ("def marker():\n# Body omitted from the observation", False),
    ("async def marker():", False),
    ("class Marker:", False),
    ("def marker():\n    return 42", True),
    ("def marker(): return 42", True),
])
def test_implementation_read_requires_body_beyond_declaration(kind, content, expected):
    success = {"tool": "read_file", "kind": kind, "value": "marker"}
    result = json.dumps({"path": "target.py", "content": content})
    assert verifies(success, "read_file", result, "lis son implémentation") == expected


@pytest.mark.parametrize("kind", ["nonempty", "contains"])
def test_implementation_body_remains_valid_when_followed_by_another_declaration(kind):
    success = {"tool": "read_file", "kind": kind, "value": "marker"}
    result = json.dumps({"path": "target.py", "start_line": 1, "end_line": 4,
                         "content": "def marker():\n    return 42\n\ndef other():"})
    assert verifies(success, "read_file", result, "lis son implémentation")


@pytest.mark.parametrize("kind", ["nonempty", "contains"])
def test_explicit_signature_read_accepts_declaration_only(kind):
    success = {"tool": "read_file", "kind": kind, "value": "marker"}
    result = json.dumps({"path": "target.py", "content": "def marker():"})
    assert verifies(success, "read_file", result, "Lis la signature de marker")


def test_reduced_declaration_alone_does_not_verify_implementation():
    result = json.dumps({"path": "target.py", "truncated": True,
                         "content_excerpts": [{"line": 1, "text": "def marker():"}]})
    assert not verifies({"tool": "read_file", "kind": "nonempty"}, "read_file", result,
                        "lis son implémentation")


def test_paths_come_from_actual_search_and_read_observations():
    source = build_tool_source("search_code", {"query": "ORION"}, json.dumps({
        "matches": [{"file": "current.py", "line": 7, "text": "ORION = 42"}],
    }))
    assert evidence_paths(source) == ["current.py"]
    read = build_tool_source("read_file", {"path": "invented.py"}, json.dumps({
        "path": "current.py", "start_line": 7, "end_line": 7, "content": "ORION = 42",
    }))
    assert evidence_paths(read) == ["current.py"]


@pytest.mark.parametrize("query", ["marker", "def marker"])
def test_search_source_identifies_exact_definition_among_observed_usages(query):
    content = json.dumps({"matches": [
        {"file": "target.py", "line": 1, "text": "def marker():"},
        {"file": "test_target.py", "line": 1, "text": "from target import marker"},
        {"file": "test_target.py", "line": 3, "text": "def test_marker():"},
        {"file": "test_target.py", "line": 4, "text": "assert marker() == 42"},
    ], "truncated": False})
    source = build_tool_source("search_code", {"query": query}, content)
    assert source["search_paths"] == ["target.py", "test_target.py"]
    assert source.get("definition_paths") == ["target.py"]
    assert source["content"].partition("Result:\n")[2] == content


def test_search_source_does_not_promote_usages_or_similar_names_to_definitions():
    content = json.dumps({"matches": [
        {"file": "usage.py", "line": 1, "text": "from target import marker"},
        {"file": "tests.py", "line": 1, "text": "def test_marker():"},
        {"file": "tests.py", "line": 2, "text": "assert marker() == 42"},
        {"file": "other.py", "line": 1, "text": "def marker_extra():"},
        {"file": "other.py", "line": 2, "text": "# def marker():"},
        {"file": "other.py", "line": 3, "text": '\"def marker():\"'},
    ]})
    source = build_tool_source("search_code", {"query": "marker"}, content)
    assert source.get("definition_paths", []) == []


def test_search_source_keeps_two_exact_definitions_ambiguous():
    content = json.dumps({"matches": [
        {"file": "one.py", "line": 1, "text": "def marker():"},
        {"file": "two.py", "line": 7, "text": "def marker():"},
    ]})
    source = build_tool_source("search_code", {"query": "marker"}, content)
    assert source.get("definition_paths") == ["one.py", "two.py"]


def test_execution_tracks_completed_remaining_budget_and_stop(monkeypatch):
    monkeypatch.setattr("app.agent.planning.perf_counter", lambda: 10.0)
    execution = PlanningExecution(GOAL, "M8_PLANNED", "dependency", max_tool_calls=3,
                                  max_total_tokens=100, max_seconds=20)
    execution.state["plan"] = validate_plan(evidence_plan(), GOAL, TOOLS, 6)
    execution.current.update(status="COMPLETED", evidence=["S1"])
    execution.tool_calls = 1
    execution.used_tokens = 40
    execution.refresh()
    assert execution.state["completed"] == ["1"]
    assert execution.state["remaining"] == ["2"]
    budgets = execution.state["budgets"]
    assert {key: budgets[key] for key in ("tool_calls", "tokens", "seconds")} == {
        "tool_calls": 2, "tokens": 60, "seconds": 20,
    }
    assert budgets["iterations"] == 5
    assert budgets["replans"] == 2
    with pytest.raises(ExecutionStopped, match="global_budget"):
        execution.check(reserve=61)
    execution.stop("goal_achieved")
    assert execution.state["stop_reason"] == "global_budget"
    assert "Objectif final non validé" in execution.summary()
    assert execution.state["events"][-1]["event"] == "stop"


def test_execution_time_budget_stops_before_next_action(monkeypatch):
    monkeypatch.setattr("app.agent.planning.perf_counter", lambda: 10.0)
    execution = PlanningExecution(GOAL, "M8_PLANNED", "dependency", max_tool_calls=3,
                                  max_total_tokens=100, max_seconds=20)
    monkeypatch.setattr("app.agent.planning.perf_counter", lambda: 30.0)
    with pytest.raises(ExecutionStopped):
        execution.check()
    assert execution.state["budgets"]["seconds"] == 0


@pytest.mark.parametrize("verb", ["Lis", "lire", "Exécute", "Execute", "Exécuter", "Executez", "Vérifie", "Verifie"])
@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_action_spans_keep_unicode_offsets_and_ignore_paths(verb, form):
    text = unicodedata.normalize(form, f"{verb} le fichier répertoire/read.py avec run_tests.")
    actions = _collection_actions(text)
    assert [match[0] for match in actions] == [unicodedata.normalize(form, verb)]
    assert text[actions[0].start():actions[0].end()] == actions[0][0]


@pytest.mark.parametrize("text", ["statut d'exécution", "execution status", "Ex?cute les tests", "read.py", "app/inspect.py", "run_tests", "lecture du résultat"])
def test_nouns_corrupted_text_and_identifiers_are_not_actions(text):
    assert not ACTION_PATTERN.findall(text)


def three_action_plan():
    goal = "Lis répertoire/read.py. Lis tests/test_read.py. Exécute les tests de tests/test_read.py avec run_tests."
    return {"goal": goal, "steps": [
        {"id": "1", "objective": "Lis répertoire/read.py", "depends_on": [],
         "success": {"tool": "read_file", "kind": "nonempty"}},
        {"id": "2", "objective": "Lis tests/test_read.py", "depends_on": [],
         "success": {"tool": "read_file", "kind": "nonempty"}},
        {"id": "3", "objective": "Exécute les tests de tests/test_read.py avec run_tests", "depends_on": ["2"],
         "success": {"tool": "run_tests", "kind": "observed"}},
    ]}


def test_independent_reads_then_dependent_tests_are_valid_without_mutation():
    raw = three_action_plan()
    before = copy.deepcopy(raw)
    validated = validate_plan(raw, raw["goal"], TOOLS, 6)
    assert raw == before
    assert [step["depends_on"] for step in validated] == [[], [], ["2"]]


def test_case_and_verb_accents_preserve_original_objective_and_file():
    raw = three_action_plan()
    raw["steps"][0]["objective"] = "lis répertoire/read.py"
    raw["steps"][2]["objective"] = "Execute les tests de tests/test_read.py avec run_tests"
    validated = validate_plan(raw, raw["goal"], TOOLS, 6)
    assert validated[0]["objective"] == "Lis répertoire/read.py"
    assert validated[2]["objective"].startswith("Exécute")
    assert raw["steps"][2]["objective"].startswith("Execute")


def test_decomposed_objective_keeps_original_path_and_action_positions():
    raw = json.loads(unicodedata.normalize("NFD", json.dumps(three_action_plan(), ensure_ascii=False)))
    validated = validate_plan(raw, raw["goal"], TOOLS, 6)
    assert validated[0]["objective"] == raw["steps"][0]["objective"]
    assert unicodedata.normalize("NFC", validated[0]["objective"]) != validated[0]["objective"]


def test_multi_action_objective_and_omitted_action_still_fail():
    raw = three_action_plan()
    raw["steps"][0]["objective"] += ". Lis tests/test_read.py"
    with pytest.raises(ValueError, match="one explicit action"):
        validate_plan(raw, raw["goal"], TOOLS, 6)
    raw = three_action_plan()
    raw["steps"].pop()
    with pytest.raises(ValueError, match="omits an explicit action"):
        validate_plan(raw, raw["goal"], TOOLS, 6)


@pytest.mark.parametrize("changes", [
    {"depends_on": ["3"]}, {"depends_on": ["missing"]}, {"depends_on": ["1", "1"]},
    {"objective": "Lis read.py"}, {"objective": "Lis répertoire/Read.py"},
    {"objective": "Lis repertoire/read.py"},
])
def test_independent_plan_still_rejects_cycles_duplicates_and_changed_paths(changes):
    raw = three_action_plan()
    raw["steps"][0].update(changes)
    with pytest.raises(ValueError, match="Step 1"):
        validate_plan(raw, raw["goal"], TOOLS, 6)


@pytest.mark.parametrize("returncode,passed,timed_out,expected", [
    (0, True, False, True), (1, False, False, False), (1, True, False, False),
    (None, True, False, False), (0, True, True, False),
])
def test_test_success_requires_consistent_real_process_status(returncode, passed, timed_out, expected):
    data = dict(returncode=returncode, passed=passed, timed_out=timed_out, stdout="pytest output")
    assert verifies({"tool": "run_tests", "kind": "nonempty"}, "run_tests", json.dumps(data)) is expected


def test_shared_read_action_keeps_both_explicit_paths_in_each_objective():
    objective = "Lis app/agent/planning.py et tests/test_planning.py"
    goal = objective + ", puis exécute les tests de tests/test_planning.py."
    raw = {"goal": goal, "steps": [
        {"id": "1", "objective": objective, "depends_on": [],
         "success": {"tool": "read_file", "kind": "nonempty", "path": "app/agent/planning.py"}},
        {"id": "2", "objective": objective, "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "path": "tests/test_planning.py"}},
        {"id": "3", "objective": "exécute les tests de tests/test_planning.py", "depends_on": ["2"],
         "success": {"tool": "run_tests", "kind": "observed"}},
    ]}
    assert len(validate_plan(raw, goal, [*TOOLS, "run_tests"], 6)) == 3
    raw["steps"][0]["objective"] = "Lis app/agent/planning.py"
    with pytest.raises(ValueError, match="omits an explicit path") as failure:
        validate_plan(raw, goal, [*TOOLS, "run_tests"], 6)
    assert f'objective="{objective}"' in str(failure.value)
    assert "success.path" in str(failure.value)


def plan(question, objectives, conditions):
    return {"goal": question, "steps": [
        {"id": str(i), "objective": objective, "depends_on": [str(i - 1)] if i > 1 else [], "success": condition}
        for i, (objective, condition) in enumerate(zip(objectives, conditions), 1)
    ]}


def test_R2_real_read_proves_complete_symbol_including_decorators(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    code = "@decorate\ndef primary():\n    value = 1\n    return value\n\ndef secondary():\n    return 2\n"
    (tmp_path / "unit.py").write_text(code, encoding="utf8")
    origin = build_tool_source("search_code", {"query": "primary"}, json.dumps(project_tools.search_code("primary")))
    condition = {"tool": "read_file", "kind": "nonempty", "from_step": "1"}
    partial = project_tools.read_file("unit.py", 1, 3)
    ast.parse(partial["content"])  # Valid Python is not proof that the function's last line was read.
    assert partial["complete_symbols"] == []
    full = project_tools.read_file("unit.py", 1, 4)
    assert full["complete_symbols"] == [{"name": "primary", "start_line": 1, "end_line": 4}]
    for read, accepted in [(partial, False), (full, True), (project_tools.read_file("unit.py", 2, 4), False),
                           (project_tools.read_file("unit.py", 6, 7), False)]:
        assert verifies(condition, "read_file", json.dumps(read), "lis son implémentation", [origin]) is accepted
    reduced = json.loads(reduce_observation(json.dumps(full), "primary", 140))
    assert "complete_symbols" not in reduced


@pytest.mark.parametrize("needle", ['return "hello"', r'path = "C:\temp"', 'first\nsecond', 'été 雪'])
def test_R9_contains_business_text(needle):
    condition = {"tool": "read_file", "kind": "contains", "value": needle}
    assert verifies(condition, "read_file", json.dumps({"content": needle}))
    assert not verifies(condition, "read_file", json.dumps({"content": "other", "path": needle}))
    assert not verifies(condition, "read_file", json.dumps({"content": "other", "note": needle}))


def test_R9_search_contains_text_not_match_metadata():
    condition = {"tool": "search_code", "kind": "contains", "value": '"hello"'}
    assert verifies(condition, "search_code", json.dumps({"matches": [{"text": 'return "hello"'}]}))
    assert not verifies(condition, "search_code", json.dumps({"matches": [{"file": '"hello"', "text": "other"}]}))


def test_R9_contains_requires_an_explicit_nonempty_expected_string():
    for value in (None, "", 1):
        assert not verifies({"tool": "read_file", "kind": "contains", "value": value},
                            "read_file", json.dumps({"content": "actual data"}))


@pytest.mark.parametrize("name,payload", [
    ("get_git_diff", {"diff": ""}), ("get_git_diff", {"diff": "+x=1"}),
    ("search_code", {"matches": [], "truncated": False}),
    ("search_code", {"matches": [{"text": "x=1"}]}),
    ("list_files", {"files": []}),
    ("run_tests", {"passed": False, "timed_out": False, "returncode": 1, "stdout": "1 failed", "stderr": ""}),
    ("run_tests", {"passed": True, "timed_out": False, "returncode": 0, "stdout": "1 passed", "stderr": ""}),
])
def test_C1_observed_negative_or_positive(name, payload):
    assert verifies({"tool": name, "kind": "observed"}, name, json.dumps(payload))


@pytest.mark.parametrize("name,payload", [
    ("get_git_diff", {"error": "failed", "diff": ""}),
    ("search_code", {"matches": [], "truncated": True}),
    ("search_code", {"matches": "not a list"}),
    ("read_file", {"path": "missing.py"}),
    ("run_tests", {"passed": False, "timed_out": True, "returncode": None}),
    ("run_tests", {"passed": False, "timed_out": False, "returncode": 3, "stdout": "internal error"}),
    ("run_tests", {"passed": True, "timed_out": False, "returncode": 1}),
])
def test_C1_execution_error_is_not_negative_evidence(name, payload):
    assert not verifies({"tool": name, "kind": "observed"}, name, json.dumps(payload))


def comparison_plan():
    goal = "Localise marker, puis compare a.py et b.py."
    data = plan(goal, ["Localise marker", "compare a.py et b.py", "compare a.py et b.py"], [
        {"tool": "search_code", "kind": "contains", "value": "marker"},
        {"tool": "read_file", "kind": "nonempty", "path": "a.py"},
        {"tool": "read_file", "kind": "nonempty", "path": "b.py"},
    ])
    data["steps"][2]["depends_on"] = ["1", "2"]
    return goal, data


def test_C3_explicit_comparison_allows_distinct_reads():
    goal, data = comparison_plan()
    assert len(validate_plan(data, goal, ["search_code", "read_file"], 6)) == 3


@pytest.mark.parametrize("mutation", ["missing", "same", "invented", "dependency", "origin"])
def test_C3_incomplete_or_invented_comparison_plan_rejected(mutation):
    goal, data = comparison_plan()
    if mutation == "missing":
        data["steps"].pop()
    elif mutation == "same":
        data["steps"][2]["success"]["path"] = "a.py"
    elif mutation == "invented":
        data["steps"][2]["success"]["path"] = "invented.py"
    elif mutation == "origin":
        data["steps"][2]["success"]["from_step"] = "unknown"
    else:
        data["steps"][2]["depends_on"] = ["1"]
    with pytest.raises(ValueError):
        validate_plan(data, goal, ["search_code", "read_file"], 6)


@pytest.mark.parametrize("target,selected,expected", [
    ("lis son implémentation", None, {"model.py"}),
    ("lis son test associé", None, {"checks.py"}),
    ("lis le fichier", None, {"model.py", "checks.py"}),
    ("lis checks.py", None, {"checks.py"}),
    ("lis invented.py", None, set()),
    ("lis le fichier", "checks.py", {"checks.py"}),
    ("lis le fichier", "invented.py", set()),
])
def test_R4_only_observed_paths_with_matching_intent(target, selected, expected):
    from app.agent.planning import observed_read_paths
    result = {"matches": [{"file": "model.py", "line": 1, "text": "class Widget:"},
                           {"file": "checks.py", "line": 1, "text": "assert Widget()"}]}
    origin = build_tool_source("search_code", {"query": "Widget"}, json.dumps(result))
    condition = {"tool": "read_file", "kind": "nonempty"}
    if selected:
        condition["path"] = selected
    assert observed_read_paths(condition, target, [origin]) == expected


def test_R11_real_window_completeness_certificate_is_still_required(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "unit.py").write_text("def previous():\n    x = 0\n    return x\n\n@decorate\ndef target():\n    x = 1\n    return x\n", encoding="utf8")
    origin = build_tool_source("search_code", {"query": "target"}, json.dumps(project_tools.search_code("target")))
    condition = {"tool": "read_file", "kind": "nonempty"}
    complete = project_tools.read_file("unit.py", 2, 8)
    assert complete["complete_symbols"] == [{"name": "target", "start_line": 5, "end_line": 8}]
    assert verifies(condition, "read_file", json.dumps(complete), "lis son implémentation", [origin])
    partial = project_tools.read_file("unit.py", 2, 7)
    assert partial["complete_symbols"] == []
    assert not verifies(condition, "read_file", json.dumps(partial), "lis son implémentation", [origin])


@pytest.mark.parametrize("stdout,answer,expected", [
    ("1 failed", "Le code de sortie vaut 1 [S1].", False),
    ("FAILED tests/test_a.py::test_wrong - AssertionError", "Le code de sortie vaut 1 [S1].", False),
    ("FAILED tests/test_a.py::test_wrong - AssertionError", "tests/test_a.py::test_wrong échoue [S1].", True),
    ("FAILED tests/test_a.py::test_wrong - AssertionError", "tests/test_a.py::test_wrong échoue [S2].", False),
])
def test_C1_exit_code_alone_does_not_identify_failing_tests(stdout, answer, expected):
    from app.agent.planning import answer_covers_actions
    source = build_tool_source("run_tests", {}, json.dumps({"returncode": 1, "stdout": stdout}))
    source["observation_id"] = "S1"
    steps = [{"id": "1", "objective": "Identifie les tests en échec", "depends_on": [],
              "success": {"tool": "run_tests", "kind": "observed"}, "evidence": ["S1"]}]
    assert answer_covers_actions(steps, [source], [source], answer) is expected


def test_C1_no_failed_tests_can_be_reported_after_successful_execution():
    from app.agent.planning import answer_covers_actions
    source = build_tool_source("run_tests", {}, json.dumps({
        "returncode": 0, "passed": True, "timed_out": False, "stdout": "1 passed"}))
    source["observation_id"] = "S1"
    steps = [{"id": "1", "objective": "Identifie les tests en échec", "depends_on": [],
              "success": {"tool": "run_tests", "kind": "observed"}, "evidence": ["S1"]}]
    assert answer_covers_actions(steps, [source], [source], "Aucun test exécuté n'a échoué [S1].")
    assert not answer_covers_actions(steps, [source], [source], "Aucun test exécuté n'a échoué [S2].")


def test_R4_multiple_associated_tests_stay_ambiguous():
    from app.agent.planning import observed_read_paths
    origin = build_tool_source("search_code", {"query": "Widget"}, json.dumps({"matches": [
        {"file": "model.py", "line": 1, "text": "class Widget:"},
        {"file": "checks.py", "line": 1, "text": "assert Widget()"},
        {"file": "other_checks.py", "line": 1, "text": "assert Widget()"},
    ]}))
    assert observed_read_paths({}, "lis le test associé à la fonction Widget", [origin]) == {"checks.py", "other_checks.py"}
