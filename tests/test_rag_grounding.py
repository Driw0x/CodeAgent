from unittest.mock import MagicMock

from app.rag.grounding import build_claim_prompt, extract_claims, verify_claim, verify_grounding


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