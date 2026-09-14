from app.rag import (
    build_context,
    build_prompt,
    format_chunk,
    format_sources,
    relative_file_path,
)
from app.rag.prompt import PROJECT_ROOT

CHUNK = {
    "file": "app/parser/file_loader.py",
    "type": "function",
    "name": "read_file",
    "content": "def read_file(path):\n    return path.read_text()",
    "start_line": 10,
    "end_line": 11,
}


def test_format_chunk():
    result = format_chunk(CHUNK)

    assert "File: app/parser/file_loader.py" in result
    assert "Lines: 10-11" in result
    assert "Type: function" in result
    assert "Name: read_file" in result
    assert "def read_file(path):" in result


def test_build_context_with_multiple_chunks():
    chunks = [CHUNK, CHUNK]

    result = build_context(chunks)

    assert result.count("File: app/parser/file_loader.py") == 2
    assert "---" in result


def test_build_prompt_contains_context_and_question():
    result = build_prompt(
        "Où est définie read_file ?",
        [CHUNK],
    )

    assert "CONTEXTE:" in result
    assert "app/parser/file_loader.py" in result
    assert "QUESTION:" in result
    assert "Où est définie read_file ?" in result


def test_relative_file_path():
    result = relative_file_path(
        str(PROJECT_ROOT / "app" / "memory" / "vector_store.py")
    )

    assert result == "app/memory/vector_store.py"


def test_format_sources():
    chunks = [
        {
            "file": str(PROJECT_ROOT / "app" / "memory" / "vector_store.py"),
            "start_line": 6,
            "end_line": 14,
        }
    ]

    result = format_sources(chunks)

    assert result == "- app/memory/vector_store.py:6-14"
