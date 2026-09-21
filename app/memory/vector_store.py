import json
from pathlib import Path

import faiss

from app.memory.embeddings import chunk_embedding_text, embeddings
from app.memory.project_state import project_relative_path

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


def build_index(model, chunks: list[dict], dimension: int = 384):
    index = faiss.IndexFlatL2(dimension)

    for chunk in chunks:
        text = chunk_embedding_text(chunk)
        vector = embeddings(model, [text])
        index.add(vector)

    return index


def update_index(
    model,
    index,
    chunks: list[dict],
    new_chunks: list[dict],
    changed_paths: list[str],
    project_path: str | Path,
):
    changed_paths = set(changed_paths)
    updated_index = faiss.IndexFlatL2(index.d)
    updated_chunks = []

    for position, chunk in enumerate(chunks):
        path = project_relative_path(chunk["file"], project_path)

        if path in changed_paths:
            continue

        vector = index.reconstruct(position).reshape(1, -1)
        updated_index.add(vector)
        updated_chunks.append(chunk)

    if new_chunks:
        texts = [chunk_embedding_text(chunk) for chunk in new_chunks]
        vectors = embeddings(model, texts)
        updated_index.add(vectors)
        updated_chunks.extend(new_chunks)

    return updated_index, updated_chunks


def index_exists(directory: str | Path) -> bool:
    directory = Path(directory)

    return (
        (directory / INDEX_FILE).exists()
        and (directory / CHUNKS_FILE).exists()
    )


def save_index(index, chunks: list[dict], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(directory / INDEX_FILE))

    (directory / CHUNKS_FILE).write_text(
        json.dumps(chunks, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_index(directory: str | Path):
    directory = Path(directory)

    index = faiss.read_index(str(directory / INDEX_FILE))

    chunks = json.loads(
        (directory / CHUNKS_FILE).read_text(encoding="utf-8")
    )

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index/chunks mismatch: {index.ntotal} vectors for {len(chunks)} chunks"
        )

    return index, chunks