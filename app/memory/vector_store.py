import json
from pathlib import Path

import faiss

from app.memory.embeddings import chunk_embedding_text, embeddings
from app.memory.project_state import project_relative_path

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
INDEX_METADATA_FILE = "index_metadata.json"
INDEX_SCHEMA_VERSION = 1
CHUNK_METADATA = {"chunk_id", "source_version", "chunk_version"}


def _validate_chunks(chunks: list[dict]) -> bool:
    annotated = [CHUNK_METADATA.issubset(chunk) for chunk in chunks]
    if any(annotated) and not all(annotated):
        raise ValueError("Chunk metadata is incomplete.")
    if annotated and all(annotated):
        identifiers = [chunk["chunk_id"] for chunk in chunks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Chunk identifiers must be unique.")
        return True
    return False


def build_index(model, chunks: list[dict], dimension: int = 384):
    index = faiss.IndexFlatL2(dimension)

    if chunks:
        vectors = embeddings(model, [chunk_embedding_text(chunk) for chunk in chunks])
        index.add(vectors)

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

    return all((directory / name).exists() for name in (INDEX_FILE, CHUNKS_FILE, INDEX_METADATA_FILE))


def save_index(index, chunks: list[dict], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if index.ntotal != len(chunks):
        raise ValueError(f"Index/chunks mismatch: {index.ntotal} vectors for {len(chunks)} chunks")
    has_chunk_metadata = _validate_chunks(chunks)

    faiss.write_index(index, str(directory / INDEX_FILE))
    (directory / CHUNKS_FILE).write_text(
        json.dumps(chunks, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (directory / INDEX_METADATA_FILE).write_text(
        json.dumps({
            "schema_version": INDEX_SCHEMA_VERSION,
            "chunk_count": len(chunks),
            "dimension": index.d,
            "chunk_metadata": has_chunk_metadata,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index(directory: str | Path):
    directory = Path(directory)
    index = faiss.read_index(str(directory / INDEX_FILE))
    chunks = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))

    if index.ntotal != len(chunks):
        raise ValueError(f"Index/chunks mismatch: {index.ntotal} vectors for {len(chunks)} chunks")

    metadata_path = directory / INDEX_METADATA_FILE
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported index schema; rebuild the index.")
        if metadata.get("chunk_count") != len(chunks) or metadata.get("dimension") != index.d:
            raise ValueError("Index metadata mismatch; rebuild the index.")
        has_chunk_metadata = _validate_chunks(chunks)
        if bool(metadata.get("chunk_metadata")) != has_chunk_metadata:
            raise ValueError("Chunk metadata contract mismatch; rebuild the index.")

    return index, chunks
