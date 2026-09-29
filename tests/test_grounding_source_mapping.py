from app.rag.grounding import extract_claims


def test_extract_claims_uses_sentence_level_citations():
    answer = (
        "LocalLLM est défini dans app/llm/local_llm.py [S1]. "
        "Le modèle par défaut est qwen2.5-coder:14b [S2]."
    )
    claims = extract_claims(answer)
    assert claims == [
        (
            "LocalLLM est défini dans app/llm/local_llm.py .",
            [1],
        ),
        (
            "Le modèle par défaut est qwen2.5-coder:14b .",
            [2],
        ),
    ]


def test_extract_claims_falls_back_to_paragraph_citations():
    answer = (
        "Le dossier contient plusieurs fichiers.\n"
        "- app/rag/pipeline.py\n"
        "- app/rag/prompt.py [S1]"
    )
    claims = extract_claims(answer)
    assert claims == [
        (
            "Le dossier contient plusieurs fichiers.",
            [1],
        ),
        (
            "- app/rag/pipeline.py\n- app/rag/prompt.py",
            [1],
        ),
    ]
