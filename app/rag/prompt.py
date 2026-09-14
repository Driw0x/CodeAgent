SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de code source.

Réponds uniquement à partir du contexte fourni.

Si l'information demandée n'est pas présente dans le contexte, indique clairement que tu ne peux pas la déterminer.

N'invente jamais de fonction, fichier, comportement ou dépendance.

Lorsque tu utilises un morceau de code, référence son fichier et ses lignes.
"""


def format_chunk(chunk: dict) -> str:
    return (
        f"File: {chunk['file']}\n"
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