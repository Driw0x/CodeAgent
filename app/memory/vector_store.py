import faiss

from app.memory.embeddings import chunk_embedding_text, embeddings


def build_index(model, chunks: list[dict], dimension: int = 384):
    index = faiss.IndexFlatL2(dimension)

    for chunk in chunks:
        text = chunk_embedding_text(chunk)
        vector = embeddings(model, [text])
        index.add(vector)

    return index