from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de code source.

Réponds uniquement à partir du contexte fourni.

Si l'information demandée n'est pas présente dans le contexte, indique clairement que tu ne peux pas la déterminer.

N'invente jamais de fonction, fichier, comportement ou dépendance.

Lorsque tu utilises un morceau de code, référence son fichier et ses lignes.
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
        source = (
            f"{relative_file_path(chunk['file'])}:"
            f"{chunk['start_line']}-{chunk['end_line']}"
        )

        if source not in seen:
            seen.add(source)
            sources.append(source)

    return "\n".join(f"- {source}" for source in sources)


def format_chunk(chunk: dict) -> str:
    return (
        f"File: {relative_file_path(chunk['file'])}\n"
        f"Lines: {chunk['start_line']}-{chunk['end_line']}\n"
        f"Type: {chunk['type']}\n"
        f"Name: {chunk['name']}\n\n"
        f"{chunk['content']}"
    )


def build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(format_chunk(chunk) for chunk in chunks)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)

    return f"""CONTEXTE:

{context}

QUESTION:

{question}
"""