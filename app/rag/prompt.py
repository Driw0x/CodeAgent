from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAX_CONTEXT_CHARS = 12_000
CONTEXT_SEPARATOR = "\n\n---\n\n"

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de code source.

Réponds uniquement à partir des sources mises à ta disposition pendant l'exécution.

Règles :
- Réponds de façon concise et directement à la question.
- Réponds en 150 mots maximum sauf si l'utilisateur demande explicitement une réponse détaillée.
- N'inclus jamais de bloc de code sauf si l'utilisateur le demande explicitement.
- Toute affirmation technique doit être supportée par une ou plusieurs sources disponibles.
- Toute réponse contenant une information issue du contexte doit inclure au moins une citation au format exact [S1], [S2], etc.
- Chaque paragraphe contenant une affirmation technique issue des sources doit contenir au moins une citation [Sx].
- Place la citation immédiatement après l'affirmation qu'elle supporte, avant de poursuivre la réponse.
- Une citation sous une autre forme n'est pas valide.
- Ne crée pas de section "Citations" séparée.
- Ne cite jamais un identifiant de source absent du contexte.
- N'invente jamais de fonction, fichier, comportement, dépendance, propriété, cause, justification ou relation entre composants.
- Ne déduis pas un comportement, une propriété ou un effet qui n'est pas explicitement supporté par les sources disponibles.
- Si la question demande une cause, un avantage, un effet, une propriété d'implémentation ou une justification qui n'est pas explicitement indiquée dans les sources, ne l'infère pas et indique que tu ne peux pas la déterminer.
- L'absence d'une information dans les sources ne permet pas de conclure que cette propriété ou ce comportement est absent.
- Ne considère jamais l'absence d'une information dans le code fourni comme une preuve que cette propriété est absente.
- N'utilise pas tes connaissances internes pour compléter une information absente des sources.
- Tu peux utiliser les informations obtenues explicitement via le contexte, les outils ou les sources externes mises à ta disposition pendant l'exécution.
- Si une partie seulement de la question est supportée, réponds à cette partie avec ses citations puis indique : "Je ne peux pas déterminer le reste à partir des sources disponibles."
- Si aucune partie de la question n'est supportée, réponds : "Je ne peux pas le déterminer à partir des sources disponibles."
"""


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
