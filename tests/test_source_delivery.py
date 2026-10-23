"""User-visible source delivery must match the complete observed target."""

import json

import pytest

from app.agent.planning import answer_code_observed
from app.agent.tool_loop import build_tool_source


CODE = "@decorate\ndef sample():\n    first = 1\n    return first"


def source(content=CODE, **metadata):
    return build_tool_source("read_file", {"path": "unit.py"}, json.dumps({
        "path": "unit.py", "start_line": 1, "end_line": 4, "content": content,
        "complete_symbols": [{"name": "sample", "start_line": 1, "end_line": 4}],
        **metadata,
    }))


@pytest.mark.parametrize("body,expected", [
    (CODE, True), ("def sample():\n    first = 1", False),
    ("def example():\n    pass", False), ("    return first", False),
])
def test_requested_complete_function_includes_decorator_and_last_line(body, expected):
    answer = f"```python\n{body}\n```\n[S1]"
    assert answer_code_observed(answer, [source()], "Montre le code exact de sample.") is expected


@pytest.mark.parametrize("metadata,expected", [
    ({"total_lines": 4}, True), ({"total_lines": 9}, False), ({}, False),
    ({"total_lines": 4, "truncated": True}, False),
])
def test_full_file_needs_actual_end_of_file(metadata, expected):
    assert answer_code_observed(f"```python\n{CODE}\n```\n[S1]", [source(**metadata)],
                                "Donne le contenu exact du fichier unit.py.") is expected


def test_requested_source_not_replaced_by_prose_or_wrong_target():
    assert not answer_code_observed("sample est définie [S1].", [source()], "Montre le code de sample.")
    assert not answer_code_observed(f"```python\n{CODE}\n```\n[S1]", [source()],
                                   "Montre le code exact de missing dans other.py.")
    assert not answer_code_observed(f"```python\n{CODE}\n```\n[S1]", [source(total_lines=4)],
                                   "Montre le code exact de missing dans unit.py.")
    assert not answer_code_observed(f"```python\n{CODE}\n```\n[S1]", [source(total_lines=4)],
                                   "Donne le contenu exact de la fonction missing dans unit.py.")


def test_complete_symbol_certificate_required_for_source_delivery():
    assert not answer_code_observed(f"```python\n{CODE}\n```\n[S1]", [source(complete_symbols=[])],
                                   "Retourne son implémentation complète : sample.")


@pytest.mark.anyio
@pytest.mark.parametrize("planning", ["auto", "off", "adaptive"])
@pytest.mark.parametrize("body,accepted", [(CODE, True), ("def sample():\n    pass", False), ("    return first", False)])
async def test_source_delivery_guard_applies_to_every_execution_mode(monkeypatch, planning, body, accepted):
    from tests.test_planning import plan
    from tests.test_planning_integration import call, make_agent, message
    from tests.test_tool_loop import tool_result
    goal = "Montre le code exact de sample. Localise sample, puis lis son implémentation."
    data = plan(goal, ["Localise sample", "lis son implémentation"], [
        {"tool": "search_code", "kind": "contains", "value": "sample"},
        {"tool": "read_file", "kind": "nonempty", "from_step": "1"}])
    calls = [call("search_code", query="sample"), call("read_file", path="unit.py")]
    responses = [message(data), *calls] if planning == "auto" else calls
    if planning == "off":
        responses.append({"role": "assistant", "content": "Terminé."})
    results = [tool_result(content={"matches": [{"file": "unit.py", "line": 2, "text": "def sample():"}]}),
               tool_result(content=json.loads(source()["content"].partition("Result:\n")[2]))]
    candidate = f"```python\n{body}\n```\n[S1]"
    agent, llm, _, _ = make_agent(monkeypatch, responses, results, generated=[candidate], planning=planning)
    agent.debug = True
    answer = await agent.ask(goal)
    assert (agent.last_grounding == "passed") is accepted
    assert len(llm.generate_calls) == 1
    assert agent.last_debug["candidate"] == candidate
    if planning != "off":
        assert (agent.last_execution["stop_reason"] == "goal_achieved") is accepted
    if not accepted:
        assert body not in answer
        assert agent.last_debug["reason"] == "source_delivery_incomplete_or_inexact"
        if planning != "off":
            assert agent.last_debug["missing_evidence"] == []
            assert all(step["retained_verified"] and not step["locally_cited"]
                       for step in agent.last_debug["verified_objectives"])


def test_exact_file_preserves_leading_and_trailing_blank_lines():
    code = "\n" + CODE + "\n"
    read = source(code, start_line=1, end_line=6, total_lines=6)
    question = "Donne le contenu exact du fichier unit.py."
    assert answer_code_observed(f"```python\n{code}\n```\n[S1]", [read], question)
    assert not answer_code_observed(f"```python\n{CODE}\n```\n[S1]", [read], question)


def test_partial_abstention_cannot_exempt_substituted_source_code():
    from app.agent.tool_loop import ToolLoop
    from tests.test_tool_loop import FakeGroundingLLM, FakeLLM
    candidate = "```python\ndef template():\n    pass\n```\n[S1]\n\nJe ne peux pas le déterminer à partir des sources disponibles."
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=FakeGroundingLLM())
    answer = agent._validate_final_answer("Montre le code exact de sample.", candidate, [source()])
    assert agent.last_grounding == "unsupported"
    assert "template" not in answer


@pytest.mark.parametrize("question,allowed", [
    ("Montre le code de cette fonction.", True),
    ("Retourne son implémentation complète.", True),
    ("Donne-moi le contenu exact de ce fichier.", True),
    ("Ne montre pas le code.", False),
    ("Do not show the code.", False),
])
def test_requested_source_formats_and_negations(question, allowed):
    from app.rag.prompt import output_contract_satisfied, requests_source_code
    assert requests_source_code(question) is allowed
    assert output_contract_satisfied(question, "```python\npass\n```") is allowed
