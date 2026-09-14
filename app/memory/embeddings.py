def embeddings(model, data):
    # The sentences to encode
    sentences = data

    # 2. Calculate embeddings by calling model.encode()
    return model.encode(sentences, show_progress_bar=False)
    # print(embeddings.shape)
    # [3, 384]

def chunk_embedding_text(chunk: dict) -> str:
    return (
        f"File: {chunk['file']}\n"
        f"Type: {chunk['type']}\n"
        f"Name: {chunk['name']}\n"
        f"Code:\n{chunk['content']}"
    )