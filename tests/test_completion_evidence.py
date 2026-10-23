"""Deterministic regressions from the recorded C and HO3 false completions."""

import pytest

from app.agent.tool_loop import build_tool_source
from app.rag import RAGPipeline
from tests.test_planning_integration import call, make_agent, message
from tests.test_tool_loop import FakeLLM, tool_result


@pytest.mark.anyio
@pytest.mark.parametrize("verdict,reference,expected", [
    ("SUPPORTED", "[S1]", "passed"),
    ("UNSUPPORTED", "[S1]", "unsupported"),
    ("SUPPORTED", "[S9]", "citation_error"),
])
async def test_compare_manifests_repair_preserves_code_without_false_completion(
    monkeypatch, verdict, reference, expected,
):
    import inspect
    from app.memory.project_state import compare_manifests

    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    code = inspect.getsource(compare_manifests)
    question = ("Montre le code. Localise la définition de compare_manifests dans app/memory/project_state.py, "
                "puis lis son implémentation et explique le résultat avec ses sources.")
    plan = {"goal": question, "steps": [
        {"id": "1", "objective": "Localise la définition de compare_manifests dans app/memory/project_state.py",
         "depends_on": [], "success": {"tool": "search_code", "kind": "contains", "value": "compare_manifests"}},
        {"id": "2", "objective": "lis son implémentation", "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "1"}},
    ]}
    path = "app/memory/project_state.py"
    candidate = f"Voici son implémentation {reference} :\n\n```python\n{code}```\n\nLes listes sont triées."
    agent, _, grounding, client = make_agent(monkeypatch, [message(plan),
        call("search_code", query="compare_manifests", path=path),
        call("read_file", path=path, start_line=45, end_line=65)], [
            tool_result(content={"matches": [{"file": path, "line": 45, "text": code.splitlines()[0]}]}),
            tool_result(content={"path": path, "start_line": 45, "end_line": 59, "content": code.rstrip("\n"),
                                 "complete_symbols": [{"name": "compare_manifests", "start_line": 45, "end_line": 59}]}),
        ], generated=[candidate])
    grounding.verdict = verdict

    answer = await agent.ask(question)

    assert len(client.calls) == 2
    assert agent.last_debug["candidate"] == candidate
    assert agent.last_debug["source_mapping"] == {"S1": "S2"}
    assert agent.last_execution["completed"] == ["1", "2"]
    assert agent.last_grounding == expected
    if expected == "passed":
        assert f"```python\n{code}```" in answer
        compile(code, "compare-manifests-regression", "exec")
        assert answer.endswith("Les listes sont triées. [S1]")
        assert agent.last_debug["citation_repair"] == "verified"
        assert agent.last_execution["stop_reason"] == "goal_achieved"
    else:
        assert "```" not in answer
        assert agent.last_execution["stop_reason"] == "insufficient_evidence"
        if expected == "citation_error":
            assert grounding.calls == []


C_QUESTION = "Localise la définition de read_dir dans app/parser/file_loader.py, puis lis le fichier retrouvé."
C_BODY = (
    "def read_dir(path):\n"
    "    p = Path(path)\n"
    "    lprog = []\n"
    "    for prog in p.glob('**/*.%s'%EXTENSIONS[LANGAGE.lower()]):\n"
    "        if not IGNORED_DIRS.intersection(prog.parts) and prog.name not in IGNORED_FILES:\n"
    '            lprog.append({"path": prog, "content": read(prog)})\n'
    "    return lprog"
)
C_CANDIDATE = (
    "La définition de la fonction `read_dir` dans le fichier `app/parser/file_loader.py` "
    "est située entre les lignes 14 et 20. Voici le contenu de cette fonction :\n\n"
    f"```python\n{C_BODY}\n```\n\n"
    "Cette fonction utilise le module `Path` pour parcourir les fichiers dans le répertoire spécifié, "
    "en filtrant les fichiers en fonction des extensions définies dans `EXTENSIONS` et en ignorant "
    "certains répertoires et fichiers spécifiés dans `IGNORED_DIRS` et `IGNORED_FILES` respectivement. "
    "Les fichiers valides sont lus et leurs chemins et contenus sont ajoutés à la liste `lprog`, "
    "qui est ensuite retournée [S1]."
)
C_CORRECT = "La fonction `read_dir` appelle `Path(path)`, filtre les fichiers et retourne `lprog` [S1]."

HO3_QUESTION = "Localise product, puis lis son implémentation, puis inspecte les tests associés dans test_product.py."
HO3_CANDIDATE = (
    "Le fichier `calc.py` contient la définition de la fonction `product` qui prend deux arguments "
    "`a` et `b` et retourne leur produit [S1]. L'implémentation de cette fonction est `return a*b` [S2]. "
    "Le fichier `test_product.py` importe la fonction `product` de `calc.py` et contient un test qui "
    "vérifie que `product(6,7)` retourne `42` [S3]."
)
# The search establishes location; the read establishes behavior. Both are cited locally.
HO3_CORRECT = (HO3_CANDIDATE.replace("produit [S1]", "produit [S1][S2]")
               .replace("`42` [S3]", "`42` [S2][S3]"))


def recorded_case(case):
    if case == "C":
        question = C_QUESTION
        objectives = ["Localise la définition de read_dir dans app/parser/file_loader.py", "lis le fichier retrouvé"]
        conditions = [{"tool": "search_code", "kind": "contains", "value": "read_dir"},
                      {"tool": "read_file", "kind": "nonempty", "from_step": "1"}]
        calls = [call("search_code", query="read_dir", path="app/parser/file_loader.py", max_results=10),
                 call("read_file", path="app/parser/file_loader.py", start_line=14, end_line=34)]
        data = [{"matches": [{"file": "app/parser/file_loader.py", "line": 14, "text": "def read_dir(path):"}]},
                {"path": "app/parser/file_loader.py", "start_line": 14, "end_line": 20, "content": C_BODY}]
    else:
        question = HO3_QUESTION
        objectives = ["Localise product", "lis son implémentation", "inspecte les tests associés dans test_product.py"]
        conditions = [{"tool": "search_code", "kind": "contains", "value": "product"},
                      {"tool": "read_file", "kind": "nonempty", "from_step": "1"},
                      {"tool": "read_file", "kind": "nonempty", "value": "test_product.py"}]
        calls = [call("search_code", query="product", max_results=10),
                 call("read_file", path="calc.py", start_line=1, end_line=51),
                 call("read_file", path="test_product.py", start_line=1, end_line=51)]
        data = [{"matches": [
                    {"file": "calc.py", "line": 1, "text": "def product(a,b):"},
                    {"file": "test_product.py", "line": 1, "text": "from calc import product"},
                    {"file": "test_product.py", "line": 2, "text": "def test_product():"},
                    {"file": "test_product.py", "line": 3, "text": "    assert product(6,7)==42"}]},
                {"path": "calc.py", "start_line": 1, "end_line": 2, "content": "def product(a,b):\n    return a*b",
                 "complete_symbols": [{"name": "product", "start_line": 1, "end_line": 2}]},
                {"path": "test_product.py", "start_line": 1, "end_line": 3,
                 "content": "from calc import product\ndef test_product():\n    assert product(6,7)==42"}]
    plan = {"goal": question, "steps": [
        {"id": str(index), "objective": objective, "depends_on": [str(index - 1)] if index > 1 else [], "success": condition}
        for index, (objective, condition) in enumerate(zip(objectives, conditions), 1)
    ]}
    return question, plan, calls, data


@pytest.mark.anyio
@pytest.mark.parametrize("case,bad,correct", [("C", C_CANDIDATE, C_CORRECT), ("HO3", HO3_CANDIDATE, HO3_CORRECT)], ids=["C", "HO3"])
@pytest.mark.parametrize("planning", ["auto", "off"])
@pytest.mark.parametrize("valid", [False, True], ids=["historical-false-success", "sufficient-evidence"])
async def test_recorded_completion_requires_locally_sufficient_evidence(monkeypatch, case, bad, correct, planning, valid):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    question, plan, calls, data = recorded_case(case)
    candidate = correct if valid else bad
    responses = [message(plan), *calls] if planning == "auto" else [*calls, {"role": "assistant", "content": "Terminé."}]
    agent, _, grounding, client = make_agent(monkeypatch, responses, [tool_result(content=item) for item in data],
                                            generated=[candidate], planning=planning)
    # A falsely optimistic verifier must not bypass the deterministic evidence checks.
    grounding.verdict = "SUPPORTED"
    answer = await agent.ask(question)
    assert len(client.calls) == len(calls)
    assert all(trace["status"] == "success" for trace in agent.last_trace)
    assert agent.last_debug["candidate"] == candidate
    assert len(agent.memory.recent[-1]["observations"]) == 2 * len(calls)
    assert agent.last_debug["source_mapping"] == ({"S1": "S2"} if case == "C" else {"S1": "S1", "S2": "S2", "S3": "S3"})
    if planning == "auto":
        assert agent.last_execution["completed"] == [step["id"] for step in plan["steps"]]
        assert agent.last_execution["remaining"] == []
        assert [step["evidence"] for step in agent.last_execution["plan"]] == [[f"S{i}"] for i in range(1, len(calls) + 1)]
        assert agent.last_execution["stop_reason"] == ("goal_achieved" if valid else "insufficient_evidence")
        assert [step["depends_on"] for step in agent.last_execution["plan"]] == [step["depends_on"] for step in plan["steps"]]
    assert agent.last_grounding == ("passed" if valid else "unsupported")
    if valid:
        assert answer == candidate
        assert grounding.calls
    else:
        assert "affirmations non supportées" in answer


@pytest.mark.anyio
async def test_planned_return_claim_cannot_borrow_a_body_cited_by_a_later_sentence(monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    question, plan, calls, data = recorded_case("HO3")
    # Correct the independent imported-function qualification, leaving the original
    # false attribution of general return behavior to the declaration-only search.
    candidate = HO3_CANDIDATE.replace("`42` [S3]", "`42` [S2][S3]")
    agent, _, grounding, _ = make_agent(monkeypatch, [message(plan), *calls],
        [tool_result(content=item) for item in data], generated=[candidate])

    answer = await agent.ask(question)
    assert agent.last_execution["completed"] == ["1", "2", "3"]
    assert agent.last_execution["stop_reason"] == "insufficient_evidence"
    assert agent.last_debug["claims"][0]["source_ids"] == [1]
    assert agent.last_debug["claims"][0]["supported"] is False
    assert grounding.calls == []
    assert "affirmations non supportées" in answer


@pytest.mark.parametrize("case,bad,correct", [("C", C_CANDIDATE, C_CORRECT), ("HO3", HO3_CANDIDATE, HO3_CORRECT)], ids=["C", "HO3"])
@pytest.mark.parametrize("valid", [False, True])
def test_rag_uses_the_same_evidence_gate(case, bad, correct, valid):
    import json

    _, _, calls, data = recorded_case(case)
    sources = [build_tool_source(call_["tool_calls"][0]["function"]["name"],
                                 call_["tool_calls"][0]["function"]["arguments"], json.dumps(item))
               for call_, item in zip(calls, data)]
    if case == "C":
        sources = sources[1:]
        # RAG has no single-source citation repair; isolate the same false type claim.
        bad = "Cette fonction utilise le module `Path` pour parcourir les fichiers [S1]."
    candidate = correct if valid else bad
    llm = FakeLLM([], generated=[candidate, *(["SUPPORTED"] * 8)])
    pipeline = RAGPipeline(retriever=lambda query, count: sources, llm=llm)
    answer = pipeline.ask("Explique le code avec ses sources.")
    if valid:
        assert answer.startswith(candidate)
    else:
        assert "affirmations non supportées" in answer
