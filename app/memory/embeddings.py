def embeddings(model, data):
    return model.encode(data, show_progress_bar=False)


def chunk_embedding_text(chunk: dict) -> str:
    return (
        f"File: {chunk['file']}\n"
        f"Type: {chunk['type']}\n"
        f"Name: {chunk['name']}\n"
        f"Code:\n{chunk['content']}"
    )
