from unittest.mock import MagicMock

import numpy as np
import pytest

from app.retrieval.faiss_retriever import FaissRetriever


CHUNKS = [
    {
        "file": "app/parser/file_loader.py",
        "type": "function",
        "name": "read_file",
        "content": "def read_file(path): ...",
        "start_line": 10,
        "end_line": 15,
    },
    {
        "file": "app/parser/file_loader.py",
        "type": "function",
        "name": "read_dir",
        "content": "def read_dir(path): ...",
        "start_line": 20,
        "end_line": 35,
    },
]


def test_retrieve_returns_matching_chunks():
    model = MagicMock()
    model.encode.return_value = np.zeros((1, 384), dtype="float32")

    index = MagicMock()
    index.search.return_value = (
        np.array([[0.1, 0.4]], dtype="float32"),
        np.array([[0, 1]]),
    )

    retriever = FaissRetriever(model, index, CHUNKS)

    results = retriever.retrieve("function that reads files", k=2)

    assert len(results) == 2
    assert results[0]["name"] == "read_file"
    assert results[1]["name"] == "read_dir"
    assert results[0]["distance"] == pytest.approx(0.1)


def test_retrieve_encodes_question():
    model = MagicMock()
    model.encode.return_value = np.zeros((1, 384), dtype="float32")

    index = MagicMock()
    index.search.return_value = (
        np.array([[0.1]], dtype="float32"),
        np.array([[0]]),
    )

    retriever = FaissRetriever(model, index, CHUNKS)

    retriever.retrieve("Where is read_file defined?", k=1)

    model.encode.assert_called_once_with(
        ["Where is read_file defined?"],
        show_progress_bar=False,
    )


def test_retrieve_limits_k_to_chunk_count():
    model = MagicMock()
    model.encode.return_value = np.zeros((1, 384), dtype="float32")

    index = MagicMock()
    index.search.return_value = (
        np.array([[0.1, 0.4]], dtype="float32"),
        np.array([[0, 1]]),
    )

    retriever = FaissRetriever(model, index, CHUNKS)

    retriever.retrieve("Question", k=10)

    index.search.assert_called_once()

    _, k = index.search.call_args.args

    assert k == 2


def test_retrieve_rejects_empty_question():
    retriever = FaissRetriever(
        MagicMock(),
        MagicMock(),
        CHUNKS,
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retriever.retrieve("")


def test_retrieve_rejects_invalid_k():
    retriever = FaissRetriever(
        MagicMock(),
        MagicMock(),
        CHUNKS,
    )

    with pytest.raises(ValueError, match="k must be greater than 0"):
        retriever.retrieve("Question", k=0)


def test_retrieve_returns_empty_list_without_chunks():
    retriever = FaissRetriever(
        MagicMock(),
        MagicMock(),
        [],
    )

    assert retriever.retrieve("Question") == []


def test_retrieve_reranks_lexically_relevant_chunk():
    chunks = [
        {
            "file": "app/rag/prompt.py",
            "type": "variable",
            "name": "SYSTEM_PROMPT",
            "content": "Tu es un assistant spécialisé dans le code.",
            "start_line": 1,
            "end_line": 10,
        },
        {
            "file": "app/memory/vector_store.py",
            "type": "function",
            "name": "build_index",
            "content": (
                "def build_index(model, chunks):\n"
                "    index = faiss.IndexFlatL2(384)\n"
                "    vector = embeddings(model, chunks)\n"
                "    index.add(vector)"
            ),
            "start_line": 6,
            "end_line": 14,
        },
    ]

    model = MagicMock()
    model.encode.return_value = np.zeros(
        (1, 384),
        dtype="float32",
    )

    index = MagicMock()
    index.search.return_value = (
        np.array([[0.5, 1.0]], dtype="float32"),
        np.array([[0, 1]]),
    )

    retriever = FaissRetriever(
        model=model,
        index=index,
        chunks=chunks,
    )

    results = retriever.retrieve(
        "Comment les embeddings sont-ils ajoutés dans FAISS ?",
        k=2,
    )

    assert results[0]["name"] == "build_index"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


def test_retrieve_reranks_behavior_chunk_above_imports():
    chunks = [
        {
            "file": "app/memory/embeddings.py",
            "type": "function",
            "name": "embeddings",
            "content": "def embeddings(model, data):\n    return model.encode(data)",
            "start_line": 3,
            "end_line": 8,
        },
        {
            "file": "app/memory/vector_store.py",
            "type": "import",
            "name": "faiss",
            "content": "import faiss",
            "start_line": 1,
            "end_line": 1,
        },
        {
            "file": "app/memory/vector_store.py",
            "type": "function",
            "name": "build_index",
            "content": (
                "def build_index(model, chunks):\n"
                "    index = faiss.IndexFlatL2(384)\n"
                "    vector = embeddings(model, chunks)\n"
                "    index.add(vector)\n"
                "    return index"
            ),
            "start_line": 6,
            "end_line": 14,
        },
    ]

    model = MagicMock()
    model.encode.return_value = np.zeros((1, 384), dtype="float32")

    index = MagicMock()
    index.search.return_value = (
        np.array([[0.8, 0.9, 1.2]], dtype="float32"),
        np.array([[0, 1, 2]]),
    )

    retriever = FaissRetriever(model, index, chunks)

    results = retriever.retrieve(
        "Comment les embeddings sont-ils ajoutés dans FAISS ?",
        k=3,
    )

    assert results[0]["name"] == "build_index"
    assert results[0]["type"] == "function"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]