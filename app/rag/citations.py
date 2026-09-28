import re

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def extract_citations(answer: str) -> list[int]:
    return [int(value) for value in CITATION_PATTERN.findall(answer)]


def is_abstention(answer: str) -> bool:
    text = answer.casefold()
    return "déterminer" in text and ("ne peux pas" in text or "ne peut pas" in text or "impossible de déterminer" in text)


def validate_citations(answer: str, source_count: int) -> bool:
    citations = extract_citations(answer)
    if not citations:
        return is_abstention(answer)
    return all(1 <= citation <= source_count for citation in citations)