from unittest.mock import MagicMock

import pytest

from app.rag import RAGPipeline


def test_ask_retrieves_chunks_and_calls_llm():
    chunks = [
        {
            "file": "app/parser/file_loader.py",
            "type": "function",
            "name": "read_file",
            "content": "def read_file(path):\n    return path.read_text()",
            "start_line": 10,
            "end_line": 11,
        }
    ]

    retriever = MagicMock(return_value=chunks)

    llm = MagicMock()
    llm.generate.return_value = (
        "La fonction read_file est définie dans "
        "app/parser/file_loader.py:10-11."
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=3,
    )

    result = pipeline.ask("Où est définie read_file ?")

    retriever.assert_called_once_with(
        "Où est définie read_file ?",
        3,
    )

    llm.generate.assert_called_once()

    assert "read_file" in result
    assert "Sources:" in result
    assert "app/parser/file_loader.py:10-11" in result


def test_ask_rejects_empty_question():
    pipeline = RAGPipeline(
        retriever=MagicMock(),
        llm=MagicMock(),
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.ask("")


def test_ask_handles_empty_retrieval():
    pipeline = RAGPipeline(
        retriever=MagicMock(return_value=[]),
        llm=MagicMock(),
    )

    result = pipeline.ask("Question")

    assert result == (
        "Aucun contexte pertinent n'a été trouvé "
        "pour répondre à cette question."
    )
