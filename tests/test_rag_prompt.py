from app.rag import (
    build_context,
    build_prompt,
    format_chunk,
    format_sources,
    relative_file_path,
)
from app.rag.prompt import PROJECT_ROOT, SYSTEM_PROMPT


CHUNK = {
    "file": "app/parser/file_loader.py",
    "type": "function",
    "name": "read_file",
    "content": "def read_file(path):\n    return path.read_text()",
    "start_line": 10,
    "end_line": 11,
}

SECOND_CHUNK = {
    "file": "app/memory/vector_store.py",
    "type": "function",
    "name": "build_index",
    "content": "def build_index(model, chunks):\n    return chunks",
    "start_line": 6,
    "end_line": 7,
}


def test_format_chunk():
    result = format_chunk(CHUNK, source_id=1)

    assert "[S1]" in result
    assert "File: app/parser/file_loader.py" in result
    assert "Lines: 10-11" in result
    assert "Type: function" in result
    assert "Name: read_file" in result
    assert "def read_file(path):" in result


def test_build_context_with_multiple_chunks():
    result = build_context([CHUNK, SECOND_CHUNK])

    assert "[S1]" in result
    assert "[S2]" in result
    assert "Name: read_file" in result
    assert "Name: build_index" in result
    assert "---" in result


def test_build_context_preserves_chunk_order():
    result = build_context([CHUNK, SECOND_CHUNK])

    assert result.index("Name: read_file") < result.index("Name: build_index")
    assert result.index("[S1]") < result.index("[S2]")


def test_build_context_removes_duplicate_chunks():
    result = build_context([CHUNK, CHUNK, SECOND_CHUNK])

    assert result.count("Name: read_file") == 1
    assert result.count("Name: build_index") == 1
    assert "[S1]" in result
    assert "[S2]" in result
    assert "[S3]" not in result


def test_build_context_respects_max_chars():
    first_section = format_chunk(CHUNK, source_id=1)
    second_section = format_chunk(SECOND_CHUNK, source_id=2)
    separator = "\n\n---\n\n"

    max_chars = (
        len(first_section)
        + len(separator)
        + len(second_section)
        - 1
    )

    result = build_context(
        [CHUNK, SECOND_CHUNK],
        max_chars=max_chars,
    )

    assert len(result) <= max_chars
    assert "[S1]" in result
    assert "[S2]" not in result


def test_build_context_skips_chunk_larger_than_budget():
    large_chunk = {
        **CHUNK,
        "content": "x" * 5000,
    }

    result = build_context(
        [large_chunk, SECOND_CHUNK],
        max_chars=500,
    )

    assert "x" * 100 not in result
    assert "Name: build_index" in result


def test_build_context_source_ids_are_contiguous():
    large_chunk = {
        **CHUNK,
        "content": "x" * 5000,
    }

    result = build_context(
        [large_chunk, SECOND_CHUNK],
        max_chars=500,
    )

    assert "[S1]" in result
    assert "[S2]" not in result


def test_build_prompt_contains_context_and_question():
    result = build_prompt(
        "Où est définie read_file ?",
        [CHUNK],
    )

    assert "CONTEXTE:" in result
    assert "[S1]" in result
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


def test_system_prompt_requires_source_grounding():
    assert "[S1]" in SYSTEM_PROMPT
    assert "supportée" in SYSTEM_PROMPT
    assert "Ne cite jamais" in SYSTEM_PROMPT


def test_system_prompt_handles_missing_information():
    assert "ne peux pas le déterminer" in SYSTEM_PROMPT
    assert "connaissances internes" in SYSTEM_PROMPT
    assert "outils" in SYSTEM_PROMPT
    assert "sources externes" in SYSTEM_PROMPT