from unittest.mock import MagicMock

import pytest

from app.rag import RAGPipeline


def make_chunk() -> dict:
    return {
        "file": "app/parser/file_loader.py",
        "type": "function",
        "name": "read_file",
        "content": "def read_file(path):\n    return path.read_text()",
        "start_line": 10,
        "end_line": 11,
    }


def test_ask_retrieves_chunks_and_calls_llm():
    chunks = [make_chunk()]
    retriever = MagicMock(return_value=chunks)
    llm = MagicMock()
    llm.generate.side_effect = [
        "La fonction read_file est définie dans app/parser/file_loader.py:10-11 [S1].",
        "SUPPORTED",
    ]
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=3)
    result = pipeline.ask("Où est définie read_file ?")
    retriever.assert_called_once_with("Où est définie read_file ?", 3)
    assert llm.generate.call_count == 2
    assert "read_file" in result
    assert "Sources:" in result
    assert "app/parser/file_loader.py:10-11" in result


def test_ask_rejects_empty_question():
    pipeline = RAGPipeline(retriever=MagicMock(), llm=MagicMock())
    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.ask("")


def test_ask_handles_empty_retrieval():
    pipeline = RAGPipeline(retriever=MagicMock(return_value=[]), llm=MagicMock())
    result = pipeline.ask("Question")
    assert result == "Aucun contexte pertinent n'a été trouvé pour répondre à cette question."


def test_ask_uses_only_chunks_selected_for_context():
    large_chunk = {
        "file": "app/large.py",
        "type": "function",
        "name": "large_function",
        "content": "x" * 15_000,
        "start_line": 1,
        "end_line": 2,
    }
    small_chunk = {
        "file": "app/small.py",
        "type": "function",
        "name": "small_function",
        "content": "def small_function():\n    return 1",
        "start_line": 5,
        "end_line": 6,
    }
    retriever = MagicMock(return_value=[large_chunk, small_chunk])
    llm = MagicMock()
    llm.generate.side_effect = ["Réponse basée sur [S1].", "SUPPORTED"]
    pipeline = RAGPipeline(retriever=retriever, llm=llm)
    result = pipeline.ask("Comment fonctionne le code ?")
    prompt = llm.generate.call_args_list[0].kwargs["prompt"]

    assert "app/large.py" not in prompt
    assert "app/small.py" in prompt
    assert "[S1]" in prompt
    assert "[S2]" not in prompt
    assert "app/large.py:1-2" not in result
    assert "app/small.py:5-6" in result


def test_ask_rejects_invalid_citation():
    llm = MagicMock()
    llm.generate.return_value = "La fonction retourne 1 [S2]."
    pipeline = RAGPipeline(retriever=MagicMock(return_value=[make_chunk()]), llm=llm)
    assert pipeline.ask("Que retourne la fonction ?") == "La réponse générée contient des citations invalides ou manquantes."
    assert llm.generate.call_count == 1


def test_ask_rejects_missing_citation():
    llm = MagicMock()
    llm.generate.return_value = "La fonction retourne 1."
    pipeline = RAGPipeline(retriever=MagicMock(return_value=[make_chunk()]), llm=llm)
    assert pipeline.ask("Que retourne la fonction ?") == "La réponse générée contient des citations invalides ou manquantes."
    assert llm.generate.call_count == 1


def test_ask_accepts_abstention_without_grounding_check():
    llm = MagicMock()
    llm.generate.return_value = "Je ne peux pas le déterminer à partir des sources disponibles."
    pipeline = RAGPipeline(retriever=MagicMock(return_value=[make_chunk()]), llm=llm)
    result = pipeline.ask("Quelle base distante est utilisée ?")
    assert "Je ne peux pas le déterminer" in result
    assert "Sources:" in result
    assert llm.generate.call_count == 1


def test_ask_checks_partial_abstention():
    llm = MagicMock()
    llm.generate.side_effect = [
        "La fonction retourne 1 [S1]. Je ne peux pas déterminer le reste à partir des sources disponibles.",
        "SUPPORTED",
    ]
    pipeline = RAGPipeline(retriever=MagicMock(return_value=[make_chunk()]), llm=llm)
    result = pipeline.ask("Que fait la fonction et utilise-t-elle un GPU ?")
    assert "Sources:" in result
    assert llm.generate.call_count == 2


def test_ask_rejects_unsupported_answer():
    llm = MagicMock()
    llm.generate.side_effect = ["Cette fonction utilise le GPU [S1].", "UNSUPPORTED"]
    pipeline = RAGPipeline(retriever=MagicMock(return_value=[make_chunk()]), llm=llm)
    assert pipeline.ask("La fonction utilise-t-elle le GPU ?") == "La réponse générée contient des affirmations non supportées par les sources."
    assert llm.generate.call_count == 2


def test_ask_rejects_answer_with_one_unsupported_claim():
    llm = MagicMock()
    llm.generate.side_effect = [
        "La fonction retourne du texte [S1]. Elle utilise le GPU [S1].",
        "SUPPORTED",
        "UNSUPPORTED",
    ]
    pipeline = RAGPipeline(retriever=MagicMock(return_value=[make_chunk()]), llm=llm)
    assert pipeline.ask("Comment fonctionne la fonction ?") == "La réponse générée contient des affirmations non supportées par les sources."
    assert llm.generate.call_count == 3
