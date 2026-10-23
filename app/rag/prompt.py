from pathlib import Path
import re

from app.rag.citations import CODE_PATTERN

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAX_CONTEXT_CHARS = 12_000
CONTEXT_SEPARATOR = "\n\n---\n\n"

CODE_BLOCK_RULE = "N'inclus jamais de bloc de code sauf si l'utilisateur le demande explicitement."

SYSTEM_PROMPT = f"""Tu es un assistant spécialisé dans l'analyse de code source.

Réponds uniquement à partir des sources mises à ta disposition pendant l'exécution.

Règles :
- Réponds de façon concise et directement à la question.
- Réponds en 150 mots maximum sauf si l'utilisateur demande explicitement une réponse détaillée.
- {CODE_BLOCK_RULE}
- Toute affirmation technique doit être supportée par une ou plusieurs sources disponibles.
- Toute réponse contenant une information issue du contexte doit inclure au moins une citation au format exact [S1], [S2], etc.
- Chaque paragraphe contenant une affirmation technique issue des sources doit contenir au moins une citation [Sx].
- Place la citation immédiatement après l'affirmation qu'elle supporte, avant de poursuivre la réponse.
- Une citation sous une autre forme n'est pas valide.
- Une demande de citations attend des références [Sx] attachées aux faits demandés, pas une liste de phrases copiées des sources. Ne fournis des extraits que si l'utilisateur les demande explicitement.
- Ne crée pas de section "Citations" séparée.
- Ne cite jamais un identifiant de source absent du contexte.
- N'invente jamais de fonction, fichier, comportement, dépendance, propriété, cause, justification ou relation entre composants.
- Ne déduis pas un comportement, une propriété ou un effet qui n'est pas explicitement supporté par les sources disponibles.
- Si la question demande une cause, un avantage, un effet, une propriété d'implémentation ou une justification qui n'est pas explicitement indiquée dans les sources, ne l'infère pas et indique que tu ne peux pas la déterminer.
- L'absence d'une information dans les sources ne permet pas de conclure que cette propriété ou ce comportement est absent.
- Ne considère jamais l'absence d'une information dans le code fourni comme une preuve que cette propriété est absente.
- N'utilise pas tes connaissances internes pour compléter une information absente des sources.
- Tu peux utiliser les informations obtenues explicitement via le contexte, les outils ou les sources externes mises à ta disposition pendant l'exécution.
- Si une partie explicite de la question n'est pas supportée, réponds aux parties supportées avec leurs citations, puis nomme précisément la partie que les sources ne permettent pas de déterminer.
- Si les sources répondent à toutes les parties demandées, conclus sans abstention sur un « reste » non demandé.
- Si aucune partie de la question n'est supportée, réponds : "Je ne peux pas le déterminer à partir des sources disponibles."
"""


def requests_source_code(question: str) -> bool:
    """Recognize explicit source delivery, excluding illustrative code requests."""
    text = question.casefold()
    if re.search(r"\b(?:sans|aucun|without|no|pas de) (?:bloc de )?code\b|\b(?:exemple|example|pseudo-code|pseudocode)\b", text):
        return False
    if re.search(r"\b(?:ne\s+|n')(?:montre\w*|affiche\w*|donne\w*|fournis\w*|retourne\w*)\s+pas\b"
                 r"|\b(?:do not|don't)\s+(?:show|display|give|provide|return)\b", text):
        return False
    return bool(re.search(
        r"\b(?:montre\w*|affiche\w*|donne\w*|fournis\w*|retourne\w*|show|display|give|provide|return)\b"
        r"[^.!?\n]{0,80}\b(?:code|implémentation|implementation|contenu|content)\b", text))


def output_contract_satisfied(question: str, answer: str) -> bool:
    """Apply CODE_BLOCK_RULE to Markdown code blocks, including quoted examples."""
    if not any(not span["inline"] for span in CODE_PATTERN.finditer(answer)):
        return True
    text = question.casefold()
    request = r"(?:montre\w*|affiche\w*|donne\w*|fournis\w*|inclu\w*|écris|show|display|give|provide|include|write)"
    if re.search(r"\b(?:sans|aucun|without|no) (?:bloc de )?code\b|\bpas de (?:bloc de )?code\b"
                 rf"|\b(?:ne\s+|n'){request}\s+pas[^.!?\n]*\b(?:code|bloc|block)\b"
                 rf"|\b(?:do not|don't)\s+{request}\b[^.!?\n]*\b(?:code|block)\b", text):
        return False
    if requests_source_code(question):
        return True
    # ponytail: explicit request forms only; unfamiliar wording cannot authorize a code block.
    return bool(re.search(
        rf"\b{request}\b"
        r"[^.!?\n]{0,80}\b(?:code|bloc|blocs|snippet|block|blocks)\b"
        r"|\b(?:avec|with) (?:un |une |an? )?(?:exemple de code|bloc de code|code example|code block)\b", text,
    ))


def relative_file_path(file: str) -> str:
    path = Path(file)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def format_sources(chunks: list[dict]) -> str:
    sources = []
    seen = set()
    for chunk in chunks:
        source = f"{relative_file_path(chunk['file'])}:" f"{chunk['start_line']}-{chunk['end_line']}"
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return "\n".join(f"- {source}" for source in sources)


def format_chunk(chunk: dict, source_id: int) -> str:
    return (
        f"[S{source_id}]\n"
        f"File: {relative_file_path(chunk['file'])}\n"
        f"Lines: {chunk['start_line']}-{chunk['end_line']}\n"
        f"Type: {chunk['type']}\n"
        f"Name: {chunk['name']}\n\n"
        f"{chunk['content']}"
    )


def chunk_key(chunk: dict) -> tuple:
    return str(chunk["file"]), chunk["start_line"], chunk["end_line"]


def unique_chunks(chunks: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for chunk in chunks:
        key = chunk_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


def select_context_chunks(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> list[dict]:
    chunks = unique_chunks(chunks)
    selected = []
    current_size = 0
    for chunk in chunks:
        source_id = len(selected) + 1
        section = format_chunk(chunk, source_id)
        separator_size = len(CONTEXT_SEPARATOR) if selected else 0
        new_size = current_size + separator_size + len(section)
        if new_size > max_chars:
            continue
        selected.append(chunk)
        current_size = new_size
    return selected


def build_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    selected = select_context_chunks(chunks, max_chars)
    return CONTEXT_SEPARATOR.join(
        format_chunk(chunk, source_id)
        for source_id, chunk in enumerate(selected, start=1)
    )


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    return f"""CONTEXTE:
    {context}
    QUESTION:
    {question}
    """
