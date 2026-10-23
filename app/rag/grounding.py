import ast
import json
import re
from pathlib import Path
from textwrap import dedent

from app.rag.citations import CODE_PATTERN, extract_citations, is_abstention, mask_code, strip_citations
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


def _code_fragments(chunk: dict) -> list[str]:
    """Keep business code separate from tool arguments and disjoint observations."""
    content = chunk.get("content", "")
    if not isinstance(content, str):
        return []
    if chunk.get("type") not in {"tool", "tool_error"}:
        return [content]
    try:
        data = json.loads(content.partition("Result:\n")[2])
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict) or chunk.get("type") == "tool_error":
        return []
    if chunk.get("name") == "read_file":
        if isinstance(data.get("content"), str):
            return [data["content"]]
        items = data.get("content_excerpts", [])
        field = "text"
    elif chunk.get("name") == "search_code":
        items = data.get("matches", [])
        field = "text"
    else:
        return []
    return [item[field] for item in items if isinstance(item, dict) and isinstance(item.get(field), str)] if isinstance(items, list) else []


_IDENTIFIER = r"[^\W\d]\w*(?:\.[^\W\d]\w*)*"
_KINDS = r"module|classe|class|fonction|function"
_QUALIFICATION = re.compile(
    rf"\b(?P<kind>{_KINDS})\s+(?:`(?P<quoted>{_IDENTIFIER})`|(?P<bare>{_IDENTIFIER}))"
    rf"|(?:`(?P<subject_quoted>{_IDENTIFIER})`|\b(?P<subject_bare>{_IDENTIFIER}))"
    rf"\s+(?:est|is)\s+(?:(?:un|une|a|an|le|la|the)\s+)?(?P<subject_kind>{_KINDS})\b",
    re.IGNORECASE,
)


def _parse_fragment(fragment: str, declarations=False):
    code = dedent(fragment).rstrip()
    if declarations and re.match(r"^(?:async\s+def|def|class)\b", code) and code.endswith(":"):
        # A visible header proves its kind, but this synthetic body must never prove behavior.
        last = code.splitlines()[-1]
        code += "\n" + " " * (len(last) - len(last.lstrip()) + 4) + "pass"
    lines = code.splitlines()
    # Keep original relative line numbers. A window can start inside the preceding body.
    # Only restart at a visible top-level declaration/decorator, never synthesize a body.
    starts = [0, *(i for i, line in enumerate(lines) if i and re.match(r"^(?:@|async\s+def\b|def\b|class\b)", line))]
    for start in starts:
        candidate = [""] * start + lines[start:]
        while len(candidate) > start:
            try:
                tree = ast.parse("\n".join(candidate))
                if tree.body:
                    return tree
                break
            except SyntaxError:
                candidate.pop()
    return None


def _qualified_definitions(nodes, prefix=""):
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = prefix + node.name
            yield node, name
            yield from _qualified_definitions(node.body, name + ".")


def _qualifications_supported(claim: str, chunks: list[dict]) -> bool:
    kinds = {"module": set(), "class": set(), "function": set()}
    observed = set()
    for chunk in chunks:
        for fragment in _code_fragments(chunk):
            tree = _parse_fragment(fragment, declarations=True)
            if tree is None:
                continue
            for node, name in _qualified_definitions(tree.body):
                kinds["class" if isinstance(node, ast.ClassDef) else "function"].add(name)
                observed.add(name)
            # ponytail: declaration evidence only; no dynamic binding or control-flow analysis.
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kinds["class" if isinstance(node, ast.ClassDef) else "function"].add(node.name)
                    observed.add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        names = {alias.name, alias.asname or alias.name.split(".")[0]}
                        kinds["module"].update(names)
                        observed.update(names)
                elif isinstance(node, ast.ImportFrom):
                    observed.update(alias.asname or alias.name for alias in node.names)
                elif isinstance(node, ast.Name):
                    observed.add(node.id)
                elif isinstance(node, ast.arg):
                    observed.add(node.arg)
    for match in _QUALIFICATION.finditer(claim):
        quoted = match["quoted"] or match["subject_quoted"]
        name = quoted or match["bare"] or match["subject_bare"]
        if not quoted and name not in observed:
            continue  # Ordinary prose, such as "La fonction retourne 1", is not a symbol claim.
        kind = (match["kind"] or match["subject_kind"]).casefold()
        kind = {"classe": "class", "fonction": "function"}.get(kind, kind)
        if name not in kinds[kind]:
            return False
    return True


def _return_evidence_present(claim: str, chunks: list[dict]) -> bool:
    # Code quoted in an answer is checked by the existing semantic verifier. This guard
    # concerns prose asserting behavior, not an occurrence of "return" inside a string.
    prose = re.sub(r"```[\s\S]*?```|`[^`]*`", "", claim)
    metadata = re.search(r"\b(?:code de sortie|exit code|returncode|stdout|stderr|sortie standard)\b", prose, re.IGNORECASE)
    python_subject = re.search(r"\b(?:fonction|function|méthode|method)\b", prose, re.IGNORECASE)
    if metadata and not python_subject:
        field = "returncode" if metadata[0].casefold() in {"code de sortie", "exit code", "returncode"} else (
            "stderr" if metadata[0].casefold() == "stderr" else "stdout")
        for chunk in chunks:
            if chunk.get("type") != "tool" or chunk.get("name") != "run_tests":
                continue
            try:
                data = json.loads(chunk["content"].partition("Result:\n")[2])
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict) and (type(data.get(field)) is int if field == "returncode" else isinstance(data.get(field), str)):
                return True  # Eligibility only: the semantic verifier still checks the value and subject.
        return False
    if not re.search(r"\b(?:retourn[^\W\d_]*|renvoi[^\W\d_]*|renvoy[^\W\d_]*|returns?)\b", prose, re.IGNORECASE):
        return True
    test_claim = re.search(r"\b(?:tests?|assertion|attendu\w*|expected)\b", prose, re.IGNORECASE)
    for chunk in chunks:
        for fragment in _code_fragments(chunk):
            tree = _parse_fragment(fragment)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                     ast.Return, ast.Yield, ast.YieldFrom, ast.Lambda)):
                    # A real body also permits semantic checking of implicit returns.
                    # Unlike the qualification check, no synthetic body is supplied here.
                    return True
                if test_claim and isinstance(node, ast.Assert):
                    return True  # An assertion can establish a test expectation, not a general return.
    return False


def _claim_paragraphs(answer: str) -> list[str]:
    paragraphs = []
    parts = []
    start = 0
    for span in CODE_PATTERN.finditer(answer):
        if span["inline"]:
            continue
        parts.extend([(answer[start:span.start()], False), (span[0], True)])
        start = span.end()
    parts.append((answer[start:], False))
    for part, fenced in parts:
        if fenced:
            if paragraphs and paragraphs[-1].endswith(":"):
                paragraphs[-1] += "\n\n" + part
            else:
                paragraphs.append(part)
        else:
            # A citation immediately following a code fence belongs to that block.
            citation = re.match(r"^[ \t]*(?:\r?\n[ \t]*)?((?:\[S\d+\][ \t]*)+[.!?]?)", part)
            if paragraphs and any(not span["inline"] for span in CODE_PATTERN.finditer(paragraphs[-1])) and citation:
                paragraphs[-1] += " " + citation[1]
                part = part[citation.end():]
            paragraphs.extend(p.strip() for p in re.split(r"\n\s*\n", part) if p.strip())
    return paragraphs


def extract_claims(answer: str) -> list[tuple[str, list[int]]]:
    claims = []

    for paragraph in _claim_paragraphs(answer):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph_source_ids = sorted(set(extract_citations(paragraph)))

        # Code and its introduction form one claim, including every literal in the block.
        boundaries = [match.end() for match in re.finditer(r"(?<=[.!?])\s+", mask_code(paragraph))]
        fenced = any(not span["inline"] for span in CODE_PATTERN.finditer(paragraph))
        sentences = [paragraph] if fenced else [paragraph[start:end] for start, end in zip([0, *boundaries], [*boundaries, len(paragraph)])]
        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_source_ids = sorted(set(extract_citations(sentence)))
            source_ids = sentence_source_ids or paragraph_source_ids
            clean_sentence = strip_citations(sentence).strip()

            if (
                clean_sentence
                and not is_pure_abstention(clean_sentence)
            ):
                claims.append((clean_sentence, source_ids))

    return claims


def claims_have_citations(answer: str) -> bool:
    claims = extract_claims(answer)
    if not claims:
        return is_abstention(answer) and not extract_citations(answer)
    return all(source_ids for _, source_ids in claims)


def build_claim_prompt(claim: str, source_ids: list[int], chunks: list[dict]) -> str:
    sources = "\n\n".join(format_chunk(chunks[source_id - 1], source_id) for source_id in source_ids)
    return (
        f"SOURCES:\n{sources}\n\n"
        f"AFFIRMATION À VÉRIFIER:\n{claim}\n\n"
        "Cette affirmation est-elle explicitement supportée par les sources ?"
    )


def verify_claim(claim: str, source_ids: list[int], chunks: list[dict], llm) -> bool:
    if not source_ids or any(source_id < 1 or source_id > len(chunks) for source_id in source_ids):
        return False
    cited = [chunks[source_id - 1] for source_id in source_ids]
    # These necessary syntax checks apply to Python evidence. Prose documentation and
    # other languages keep the existing verifier; they need not contain Python nodes.
    if all(Path(str(chunk.get("file", ""))).suffix.casefold() in {"", ".py", ".pyi"} for chunk in cited):
        if not _qualifications_supported(claim, cited) or not _return_evidence_present(claim, cited):
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


def is_pure_abstention(sentence: str) -> bool:
    """Exempt only simple absence statements; mixed or unfamiliar clauses are checked."""
    text = sentence.casefold().strip().rstrip(".!?").strip()
    text = re.sub(r"^cependant,\s*", "", text)
    if CODE_PATTERN.search(text) or re.search(r"\b(?:et|mais|or|car|donc|pourtant|tandis|alors|and|but|while|whereas)\b|[,;:=]", text):
        return False
    return bool(re.fullmatch(
        r"(?:"
        r"(?:je ne peux pas|on ne peut pas|il est impossible de) (?:le )?déterminer(?: .+)?"
        r"|impossible de déterminer(?: .+)?"
        r"|(?:(?:le|ce) (?:document|code|contexte|fichier)|les sources(?: disponibles)?) "
        r"ne fourni(?:t|ssent) (?:aucune information|pas d'information)(?: .+)?"
        r"|[^.!?]+ n'est pas mentionné(?:e)?(?: .+)?"
        r")", text,
    ))
