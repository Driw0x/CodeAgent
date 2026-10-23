import json
from unittest.mock import MagicMock

import pytest

from app.agent.tool_loop import ToolLoop, build_tool_source
from app.rag.citations import extract_citations
from app.rag.grounding import build_claim_prompt, extract_claims, verify_claim, verify_grounding
from tests.test_tool_loop import FakeLLM


def make_chunk(name: str, content: str) -> dict:
    return {
        "file": f"app/{name}.py",
        "type": "function",
        "name": name,
        "content": content,
        "start_line": 1,
        "end_line": 2,
    }


def test_extract_claims():
    answer = "La fonction sauvegarde l'index [S1]. Elle sauvegarde aussi les chunks [S1]."
    claims = extract_claims(answer)
    assert claims == [
        ("La fonction sauvegarde l'index .", [1]),
        ("Elle sauvegarde aussi les chunks .", [1]),
    ]


def test_extract_claims_uses_paragraph_citations():
    answer = "La fonction sauvegarde l'index. Elle sauvegarde les chunks [S1]."
    claims = extract_claims(answer)
    assert claims[0][1] == [1]
    assert claims[1][1] == [1]


def test_extract_claims_ignores_abstention():
    answer = "La fonction sauvegarde l'index [S1].\n\nJe ne peux pas déterminer le reste à partir des sources disponibles."
    claims = extract_claims(answer)
    assert len(claims) == 1
    assert claims[0][1] == [1]


def test_build_claim_prompt_uses_cited_source():
    chunks = [
        make_chunk("first", "def first():\n    return 1"),
        make_chunk("second", "def second():\n    return 2"),
    ]
    prompt = build_claim_prompt("La fonction retourne 2.", [2], chunks)
    assert "[S2]" in prompt
    assert "def second" in prompt
    assert "def first" not in prompt


def test_verify_claim_supported():
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"
    chunks = [make_chunk("test", "def test():\n    return 1")]
    assert verify_claim("La fonction retourne 1.", [1], chunks, llm)


def test_verify_claim_unsupported():
    llm = MagicMock()
    llm.generate.return_value = "UNSUPPORTED"
    chunks = [make_chunk("test", "def test():\n    return 1")]
    assert not verify_claim("La fonction utilise un GPU.", [1], chunks, llm)


def test_verify_claim_rejects_missing_source():
    llm = MagicMock()
    chunks = [make_chunk("test", "def test():\n    return 1")]
    assert not verify_claim("La fonction retourne 1.", [], chunks, llm)
    llm.generate.assert_not_called()


def test_verify_grounding_supported():
    llm = MagicMock()
    llm.generate.side_effect = ["SUPPORTED", "SUPPORTED"]
    chunks = [make_chunk("test", "def test():\n    return 1")]
    answer = "La fonction test existe [S1]. Elle retourne 1 [S1]."
    assert verify_grounding(answer, chunks, llm)
    assert llm.generate.call_count == 2


def test_verify_grounding_rejects_one_unsupported_claim():
    llm = MagicMock()
    llm.generate.side_effect = ["SUPPORTED", "UNSUPPORTED"]
    chunks = [make_chunk("test", "def test():\n    return 1")]
    answer = "La fonction retourne 1 [S1]. Elle utilise le GPU [S1]."
    assert not verify_grounding(answer, chunks, llm)
    assert llm.generate.call_count == 2


def test_verify_grounding_rejects_uncited_claim():
    llm = MagicMock()
    chunks = [make_chunk("test", "def test():\n    return 1")]
    assert not verify_grounding("La fonction retourne 1.", chunks, llm)


def test_verify_grounding_preserves_llm_stats():
    llm = MagicMock()
    llm.last_stats = {"total_tokens": 42}

    def generate(**kwargs):
        llm.last_stats = {"total_tokens": 10}
        return "SUPPORTED"
    llm.generate.side_effect = generate
    chunks = [make_chunk("test", "def test():\n    return 1")]
    assert verify_grounding("La fonction retourne 1 [S1].", chunks, llm)
    assert llm.last_stats == {"total_tokens": 42}


def test_extract_claims_ignores_missing_information():
    answer = "La fonction sauvegarde les données [S1]. Cependant, le code ne fournit aucune information sur leur durée de conservation. Je ne peux pas déterminer le reste à partir des sources disponibles."
    claims = extract_claims(answer)
    assert len(claims) == 1
    assert "sauvegarde les données" in claims[0][0]


def test_extract_claims_keeps_negative_technical_claim():
    answer = "Cette opération n'est pas parallélisée sur le GPU [S1]."
    claims = extract_claims(answer)
    assert len(claims) == 1


def test_extract_claims_keeps_code_and_its_introduction_together():
    answer = "Voici son implémentation :\n\n```python\ndef marker():\n    return 42\n``` [S1]"
    claims = extract_claims(answer)
    assert len(claims) == 1
    assert "Voici son implémentation" in claims[0][0]
    assert "return 42" in claims[0][0]
    assert claims[0][1] == [1]


def test_extract_claims_code_inherits_only_its_introducing_paragraph_citations():
    answer = (
        "La déclaration est à la ligne 1 [S1].\n\n"
        "Voici le corps lu [S2] :\n\n```python\ndef marker():\n\n    return 42\n```\n\n"
        "Cette fonction retourne un entier [S2]."
    )
    claims = extract_claims(answer)
    code = next((claim, ids) for claim, ids in claims if "```" in claim)
    assert code[1] == [2]
    assert "\n\n    return 42" in code[0]
    assert claims[0][1] == [1]


def test_extract_claims_does_not_give_unintroduced_code_another_paragraphs_citation():
    answer = "Une fonction existe [S1].\n\n```python\ndef invented():\n    return 99\n```"
    assert extract_claims(answer)[-1][1] == []


def test_grounding_still_checks_incorrect_code_after_its_introduction():
    llm = MagicMock()
    llm.generate.return_value = "UNSUPPORTED"
    answer = "Voici son implémentation :\n\n```python\ndef marker():\n    return 99\n``` [S1]"
    assert not verify_grounding(answer, [make_chunk("marker", "def marker():\n    return 42")], llm)
    assert "return 99" in llm.generate.call_args.kwargs["prompt"]


def test_code_does_not_take_a_citation_from_the_next_paragraph():
    answer = "```python\nreturn 99\n```\n\n[S2] Le second fichier existe [S2]."
    claims = extract_claims(answer)
    assert claims[0][1] == []
    assert claims[1][1] == [2]


def test_two_code_blocks_keep_distinct_local_sources():
    answer = (
        "Premier corps [S1] :\n\n```python\nreturn 1\n```\n\n"
        "Second corps [S2] :\n\n```python\nreturn 2\n```\n[S2]."
    )
    claims = extract_claims(answer)
    assert len(claims) == 2
    assert [ids for _, ids in claims] == [[1], [2]]


def source(code, path="sample.py"):
    return build_tool_source("read_file", {"path": path}, json.dumps({"path": path, "content": code}))


@pytest.mark.parametrize("code", ['print("[S1]")', 'print("[S2]")', 'items = ["[S17]", "`[S3]`"]'])
def test_R6_code_literals_reach_verifier_unchanged(code):
    candidate = f"Code [S1] :\n\n```python\n{code}\n```"
    judge = MagicMock()
    judge.generate.return_value = "SUPPORTED"
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=judge)
    assert agent._finalize_answer(candidate, [source(code)]) == candidate
    assert extract_citations(candidate) == [1]
    assert extract_claims(candidate) == [(f"Code  :\n\n```python\n{code}\n```", [1])]
    claim = judge.generate.call_args.kwargs["prompt"].split("AFFIRMATION À VÉRIFIER:\n", 1)[1]
    assert f"```python\n{code}\n```" in claim


def test_R6_incorrect_literal_is_rejected_and_inline_is_preserved():
    judge = MagicMock()
    judge.generate.side_effect = lambda **kw: "UNSUPPORTED" if 'print("[S1]")' in kw["prompt"].split("AFFIRMATION À VÉRIFIER:")[1] else "SUPPORTED"
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=judge)
    candidate = 'Code [S1] :\n\n```python\nprint("[S1]")\n```'
    assert agent._finalize_answer(candidate, [source('print("")')]) != candidate
    assert agent.last_grounding == "unsupported"
    inline = 'Littéral `print("[S8]")` [S1].'
    assert extract_claims(inline) == [('Littéral `print("[S8]")` .', [1])]
    assert agent._finalize_answer(inline, [source('print("[S8]")')]) == inline
    assert agent._finalize_answer(inline + " Erreur [S9].", [source("")]) != inline
    assert agent.last_grounding == "citation_error"


def test_R6_multiple_claims_keep_all_code_and_local_sources():
    candidate = ('Premier [S1] :\n\n```python\nprint("[S9]")\n```\n\n'
                 'Second [S2] :\n\n~~~python\nprint("[S7]")\n~~~')
    expected = [('Premier  :\n\n```python\nprint("[S9]")\n```', [1]),
                ('Second  :\n\n~~~python\nprint("[S7]")\n~~~', [2])]
    assert extract_claims(candidate) == expected
    judge = MagicMock()
    judge.generate.return_value = "SUPPORTED"
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=judge)
    assert agent._finalize_answer(candidate, [source('print("[S9]")'), source('print("[S7]")')]) == candidate
    assert [call.kwargs["prompt"].split("AFFIRMATION À VÉRIFIER:\n")[1].split("\n\nCette affirmation")[0]
            for call in judge.generate.call_args_list] == [claim for claim, _ in expected]
    assert extract_citations('    print("[S8]")\n\n[S1]') == [1]
    assert extract_claims('Valeur `"[S9]. autre"` [S1].') == [('Valeur `"[S9]. autre"` .', [1])]


@pytest.mark.parametrize("prefix", [
    "Le document ne fournit aucune information sur la mémoire",
    "Je ne peux pas déterminer la durée",
    "La durée n'est pas mentionnée",
])
@pytest.mark.parametrize("value", [1, 99])
def test_R7_mixed_abstention_is_verified(prefix, value):
    candidate = f"La valeur vaut 1 [S1]. {prefix} et la valeur vaut {value} [S1]."
    judge = MagicMock()
    judge.generate.side_effect = lambda **kw: "UNSUPPORTED" if "99" in kw["prompt"].split("AFFIRMATION À VÉRIFIER:")[1] else "SUPPORTED"
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=judge)
    answer = agent._finalize_answer(candidate, [source("VALUE = 1")])
    assert judge.generate.call_count == 2
    assert f"la valeur vaut {value}" in judge.generate.call_args.kwargs["prompt"]
    assert agent.last_grounding == ("passed" if value == 1 else "unsupported")
    assert (answer == candidate) is (value == 1)


def test_R7_pure_abstention_and_normal_facts():
    pure = "Je ne peux pas le déterminer à partir des sources disponibles."
    assert extract_claims(pure) == []
    assert extract_claims("Le document ne fournit aucune information sur la durée.") == []
    assert len(extract_claims("VALUE vaut 1 [S1]. VALUE vaut 99 [S1].")) == 2


@pytest.mark.parametrize("join", [", la capacité vaut 99", "; la capacité vaut 99", " tandis que la capacité vaut 99"])
def test_R7_absence_cannot_hide_a_second_clause(join):
    answer = "Le document ne fournit aucune information sur la durée" + join + " [S1]."
    assert extract_claims(answer) == [(answer.replace("[S1]", "").strip(), [1])]
