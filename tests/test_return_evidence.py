import json
from unittest.mock import MagicMock

import pytest

from app.rag.grounding import verify_claim, verify_grounding

from app.agent.tool_loop import build_tool_source
from tests.test_tool_loop import FakeGroundingLLM


def tool_source(name, result, *, file="arithmetic.py", arguments=None):
    return {
        "file": file,
        "type": "tool",
        "name": name,
        "start_line": 1,
        "end_line": 3,
        "content": f"Tool: {name}\nArguments: {json.dumps(arguments or {})}\nResult:\n{json.dumps(result)}",
    }


def supporting_model():
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"
    return llm


def test_general_return_cannot_borrow_an_uncited_body_or_test_assertion():
    search = tool_source("search_code", {"matches": [
        {"file": "arithmetic.py", "line": 1, "text": "def combine(a, b):"},
        {"file": "test_arithmetic.py", "line": 1, "text": "from arithmetic import combine"},
        {"file": "test_arithmetic.py", "line": 3, "text": "assert combine(2, 3) == 6"},
    ]})
    body = tool_source("read_file", {"content": "def combine(a, b):\n    return a * b"})
    llm = supporting_model()
    assert not verify_claim("combine retourne le produit de a et b.", [1], [search, body], llm)
    llm.generate.assert_not_called()


@pytest.mark.parametrize("text", [
    "def combine(a, b):",
    "from arithmetic import combine",
    "assert combine(2, 3) == 6",
    "# return a * b",
    'description = "return a * b"',
])
def test_non_operational_search_lines_cannot_prove_a_general_return(text):
    source = tool_source("search_code", {"matches": [{"text": text}]})
    llm = supporting_model()
    assert not verify_claim("combine retourne un résultat.", [1], [source], llm)
    llm.generate.assert_not_called()


def test_tool_arguments_and_metadata_are_not_return_evidence():
    source = tool_source("search_code", {"matches": [{"text": "def combine(a, b):"}]},
                         arguments={"query": "return a * b"})
    llm = supporting_model()
    assert not verify_claim("combine retourne leur produit.", [1], [source], llm)
    llm.generate.assert_not_called()


def test_declaration_only_read_cannot_prove_a_return():
    source = tool_source("read_file", {"content": "def combine(a, b):"})
    llm = supporting_model()
    assert not verify_claim("combine retourne leur produit.", [1], [source], llm)
    llm.generate.assert_not_called()


@pytest.mark.parametrize("name,result,claim", [
    ("read_file", {"content": "def combine(a, b):\n    return a * b"},
     "combine retourne leur produit."),
    ("read_file", {"content_excerpts": [{"line": 2, "text": "    return a * b"}]},
     "combine retourne leur produit."),
    ("search_code", {"matches": [{"text": "    return a * b"}]},
     "combine retourne leur produit."),
    ("search_code", {"matches": [{"text": "def combine(a, b): return a * b"}]},
     "combine retourne leur produit."),
    ("read_file", {"content": "combine = lambda a, b: a * b"},
     "combine retourne leur produit."),
    ("read_file", {"content": "def combine(a, b):\n    yield a * b"},
     "combine retourne un générateur."),
])
def test_operational_fragments_remain_eligible_for_model_verification(name, result, claim):
    source = tool_source(name, result)
    llm = supporting_model()
    assert verify_claim(claim, [1], [source], llm)
    llm.generate.assert_called_once()


def test_explicit_test_claim_can_use_its_assertion():
    source = tool_source("read_file", {"content": (
        "from arithmetic import combine\n"
        "def test_combine():\n"
        "    assert combine(2, 3) == 6"
    )}, file="test_arithmetic.py")
    llm = supporting_model()
    assert verify_claim("Le test vérifie que combine(2, 3) retourne 6.", [1], [source], llm)
    llm.generate.assert_called_once()


def test_search_declaration_still_supports_a_location_claim():
    source = tool_source("search_code", {"matches": [
        {"file": "arithmetic.py", "line": 1, "text": "def combine(a, b):"},
    ]})
    llm = supporting_model()
    assert verify_claim("combine est défini dans arithmetic.py à la ligne 1.", [1], [source], llm)
    llm.generate.assert_called_once()


def test_bare_identifier_is_not_mistaken_for_a_return_verb():
    source = tool_source("search_code", {"matches": [{"text": "def retourne_resultat():"}]})
    llm = supporting_model()
    assert verify_claim("La fonction retourne_resultat est déclarée.", [1], [source], llm)
    llm.generate.assert_called_once()


def test_complete_body_can_support_an_implicit_return():
    source = tool_source("read_file", {"content": "def nothing():\n    pass"})
    llm = supporting_model()
    assert verify_claim("nothing retourne None.", [1], [source], llm)
    llm.generate.assert_called_once()


def test_complete_body_is_not_lost_when_the_following_declaration_is_cut():
    source = tool_source("read_file", {"content": "def combine(a,b):\n    return a*b\n\ndef following():"})
    llm = supporting_model()
    assert verify_claim("combine retourne le produit de a et b.", [1], [source], llm)
    llm.generate.assert_called_once()


def test_explicit_documentation_does_not_require_a_python_return_node():
    source = tool_source("read_file", {"content": "combine retourne le produit de ses arguments."},
                         file="README.md")
    llm = supporting_model()
    assert verify_claim("combine retourne le produit de ses arguments.", [1], [source], llm)
    llm.generate.assert_called_once()


def test_return_evidence_does_not_override_a_negative_model_verdict():
    source = tool_source("read_file", {"content": "def combine(a, b):\n    return a * b"})
    llm = supporting_model()
    llm.generate.return_value = "UNSUPPORTED"
    assert not verify_claim("combine retourne leur produit.", [1], [source], llm)
    llm.generate.assert_called_once()


def test_answer_verification_uses_the_same_local_return_guard():
    search = tool_source("search_code", {"matches": [{"text": "def combine(a, b):"}]})
    body = tool_source("read_file", {"content": "def combine(a, b):\n    return a * b"})
    llm = supporting_model()
    answer = "combine retourne leur produit [S1]. Son corps contient return a * b [S2]."
    assert not verify_grounding(answer, [search, body], llm)
    llm.generate.assert_not_called()


@pytest.mark.anyio
async def test_planned_final_cannot_borrow_another_claims_body_citation(monkeypatch):
    from tests.test_planning_integration import call, make_agent, message
    from tests.test_tool_loop import tool_result

    goal = "Localise combine, puis lis son implémentation, puis inspecte les tests associés dans test_arithmetic.py."
    objectives = goal.removesuffix(".").split(", puis ")
    plan = {"goal": goal, "steps": [
        {"id": "1", "objective": objectives[0], "depends_on": [],
         "success": {"tool": "search_code", "kind": "contains", "value": "combine"}},
        {"id": "2", "objective": objectives[1], "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "1"}},
        {"id": "3", "objective": objectives[2], "depends_on": ["2"],
         "success": {"tool": "read_file", "kind": "nonempty"}},
    ]}
    candidate = (
        "La fonction `combine` prend a et b et retourne leur produit [S1]. "
        "Son implémentation contient `return a * b` [S2]. "
        "Le test importe la fonction `combine` et vérifie que combine(2, 3) retourne 6 [S2][S3]."
    )
    results = [
        tool_result(content={"matches": [
            {"file": "arithmetic.py", "line": 1, "text": "def combine(a, b):"},
            {"file": "test_arithmetic.py", "line": 1, "text": "from arithmetic import combine"},
            {"file": "test_arithmetic.py", "line": 3, "text": "assert combine(2, 3) == 6"},
        ]}),
        tool_result(content={"path": "arithmetic.py", "start_line": 1, "end_line": 2,
                             "content": "def combine(a, b):\n    return a * b",
                             "complete_symbols": [{"name": "combine", "start_line": 1, "end_line": 2}]}),
        tool_result(content={"path": "test_arithmetic.py", "start_line": 1, "end_line": 3,
                             "content": "from arithmetic import combine\ndef test_combine():\n    assert combine(2, 3) == 6"}),
    ]
    agent, _, grounding, client = make_agent(monkeypatch, [
        message(plan), call("search_code", query="combine"),
        call("read_file", path="arithmetic.py"), call("read_file", path="test_arithmetic.py"),
    ], results, generated=[candidate])
    answer = await agent.ask(goal)
    assert len(client.calls) == 3
    assert agent.last_execution["completed"] == ["1", "2", "3"]
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert agent.last_grounding == "unsupported"
    assert "Objectif final non validé" in answer
    assert grounding.calls == []


@pytest.mark.parametrize("claim", ["L'exécution retourne un code de sortie 0.", "Le code de sortie vaut 0.",
                                       "L'outil retourne un stdout vide."])
def test_R8_metadata_reaches_semantic_grounding(claim):
    source = build_tool_source("run_tests", {}, json.dumps({"returncode": 0, "stdout": "", "stderr": ""}))
    judge = FakeGroundingLLM()
    assert verify_claim(claim, [1], [source], judge)
    assert len(judge.calls) == 1
    judge.verdict = "UNSUPPORTED"
    assert not verify_claim(claim, [1], [source], judge)


@pytest.mark.parametrize("payload", [{"stdout": "returncode: 0"}, {"returncode": None}, {}])
def test_R8_absent_metadata_cannot_bypass_return_guard(payload):
    judge = FakeGroundingLLM()
    source = build_tool_source("run_tests", {}, json.dumps(payload))
    assert not verify_claim("L'exécution retourne un code de sortie 0.", [1], [source], judge)
    assert not judge.calls


def test_R8_function_claim_cannot_use_tool_metadata():
    source = build_tool_source("run_tests", {}, json.dumps({"returncode": 0}))
    assert not verify_claim("La fonction `target` retourne 0.", [1], [source], FakeGroundingLLM())


@pytest.mark.parametrize("claim", ["Le code de sortie vaut 0.", "L'exécution retourne un code de sortie 0."])
def test_R8_absent_exit_code_rejected_for_equivalent_wording(claim):
    source = build_tool_source("run_tests", {}, json.dumps({"stdout": "returncode: 0"}))
    judge = FakeGroundingLLM()
    assert not verify_claim(claim, [1], [source], judge)
    assert not judge.calls
