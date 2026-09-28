import re

from app.rag.citations import extract_citations, is_abstention
from app.rag.prompt import format_chunk

GROUNDING_SYSTEM_PROMPT = """Tu vérifies si une affirmation est strictement supportée par les sources fournies.

Une affirmation est SUPPORTED seulement si son contenu technique ou factuel est explicitement établi par les sources.

Règles :
- Une affirmation plausible mais non explicitement supportée est UNSUPPORTED.
- Une déduction, explication, cause, avantage, effet ou justification absente des sources est UNSUPPORTED.
- L'absence d'une information dans les sources ne prouve jamais son contraire.
- Si les sources ne mentionnent pas une propriété, l'affirmation ne peut pas déclarer que cette propriété est présente ou absente.
- Une citation correcte ne suffit pas : le contenu de la source doit réellement supporter l'affirmation.
- Au moindre doute sur le support explicite de l'affirmation, réponds UNSUPPORTED.
- Réponds uniquement par SUPPORTED ou UNSUPPORTED.

Exemples :
Source : une fonction sauvegarde des données au format JSON.
Affirmation : "Le format JSON améliore les performances."
Verdict : UNSUPPORTED

Source : le code effectue une opération sans mentionner le GPU.
Affirmation : "Cette opération n'est pas parallélisée sur le GPU."
Verdict : UNSUPPORTED
"""


def extract_claims(answer: str) -> list[tuple[str, list[int]]]:
    claims = []

    for paragraph in answer.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        source_ids = sorted(set(extract_citations(paragraph)))
        clean_paragraph = re.sub(r"\[S\d+\]", "", paragraph).strip()

        for sentence in re.split(r"(?<=[.!?])\s+", clean_paragraph):
            sentence = sentence.strip()
            if sentence and not is_abstention(sentence) and not is_missing_information(sentence):
                claims.append((sentence, source_ids))

    return claims


def build_claim_prompt(claim: str, source_ids: list[int], chunks: list[dict]) -> str:
    sources = "\n\n".join(format_chunk(chunks[source_id - 1], source_id) for source_id in source_ids)
    return f"SOURCES:\n{sources}\n\nAFFIRMATION À VÉRIFIER:\n{claim}\n\nCette affirmation est-elle explicitement supportée par les sources ?"


def verify_claim(claim: str, source_ids: list[int], chunks: list[dict], llm) -> bool:
    if not source_ids or any(source_id < 1 or source_id > len(chunks) for source_id in source_ids):
        return False

    prompt = build_claim_prompt(claim, source_ids, chunks)
    previous_stats = getattr(llm, "last_stats", None)
    verdict = llm.generate(prompt=prompt, system_prompt=GROUNDING_SYSTEM_PROMPT).strip().upper()

    if hasattr(llm, "last_stats"):
        llm.last_stats = previous_stats

    return verdict == "SUPPORTED"


def verify_grounding(answer: str, chunks: list[dict], llm) -> bool:
    claims = extract_claims(answer)

    if not claims:
        return False

    return all(verify_claim(claim, source_ids, chunks, llm) for claim, source_ids in claims)


def is_missing_information(sentence: str) -> bool:
    text = sentence.casefold()
    return (
        "aucune information" in text
        or "n'est pas mentionné" in text
        or "n'est pas mentionnée" in text
        or "ne fournit pas d'information" in text
        or "ne fournit aucune information" in text
    )