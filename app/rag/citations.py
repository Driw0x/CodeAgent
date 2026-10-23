import re

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
# Shared with citation repair: closed fences, inline spans, and unsafe open spans.
CODE_PATTERN = re.compile(
    r"(?m:^[ \t]{0,3}(?:>[ \t]?)*(?P<ticks>`{3,})[^\r\n]*\r?\n)[\s\S]*?"
    r"(?:(?m:^[ \t]{0,3}(?:>[ \t]?)*(?P<ticks_end>(?P=ticks)`*)(?=[ \t]*(?:\r?$|\[S\d+\])))|\Z)"
    r"|(?m:^[ \t]{0,3}(?:>[ \t]?)*(?P<tildes>~{3,})[^\r\n]*\r?\n)[\s\S]*?"
    r"(?:(?m:^[ \t]{0,3}(?:>[ \t]?)*(?P<tildes_end>(?P=tildes)~*)(?=[ \t]*(?:\r?$|\[S\d+\])))|\Z)"
    r"|(?P<indented>(?m:^(?: {4}|\t)[^\r\n]*(?:\r?\n(?: {4}|\t)[^\r\n]*)*))"
    r"|(?P<inline>`+)[\s\S]*?(?:(?P<inline_end>(?<!`)(?P=inline))(?!`)|\Z)"
)


def citation_matches(answer: str):
    spans = list(CODE_PATTERN.finditer(answer))
    return [match for match in CITATION_PATTERN.finditer(answer)
            if not any(span.start() <= match.start() < span.end() for span in spans)]


def strip_citations(answer: str) -> str:
    for match in reversed(citation_matches(answer)):
        answer = answer[:match.start()] + answer[match.end():]
    return answer


def mask_code(answer: str) -> str:
    for span in reversed(list(CODE_PATTERN.finditer(answer))):
        answer = answer[:span.start()] + " " * (span.end() - span.start()) + answer[span.end():]
    return answer


def extract_citations(answer: str) -> list[int]:
    return [int(match[1]) for match in citation_matches(answer)]


def is_abstention(answer: str) -> bool:
    text = answer.casefold()
    return "déterminer" in text and ("ne peux pas" in text or "ne peut pas" in text or "impossible de déterminer" in text)


def validate_citations(answer: str, source_count: int) -> bool:
    citations = extract_citations(answer)
    if not citations:
        return is_abstention(answer)
    return all(1 <= citation <= source_count for citation in citations)
