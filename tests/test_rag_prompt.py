from app.rag.prompt import build_context, build_prompt, format_chunk


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