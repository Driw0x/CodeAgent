from app.agent.tool_loop import (
    consolidate_sources,
    extract_search_hints,
    optimize_tool_arguments,
)


def test_extract_search_hints():
    content = (
        "app/llm/local_llm.py:12:class LocalLLM:\n"
        "app/llm/local_llm.py:20:def generate(...):"
    )
    assert extract_search_hints(content) == [
        ("app/llm/local_llm.py", 12),
        ("app/llm/local_llm.py", 20),
    ]


def test_search_code_defaults_to_ten_results():
    arguments = optimize_tool_arguments(
        name="search_code",
        arguments={"query": "LocalLLM", "path": "."},
        question="Trouve LocalLLM.",
        search_hints={},
    )
    assert arguments["max_results"] == 10


def test_read_file_uses_search_window():
    arguments = optimize_tool_arguments(
        name="read_file",
        arguments={"path": "app/llm/local_llm.py"},
        question=(
            "Trouve LocalLLM puis lis le fichier correspondant."
        ),
        search_hints={"app/llm/local_llm.py": [20]},
    )
    assert arguments == {"path": "app/llm/local_llm.py", "start_line": 5, "end_line": 55}


def test_read_file_keeps_full_file_when_requested():
    arguments = optimize_tool_arguments(
        name="read_file",
        arguments={"path": "app/llm/local_llm.py"},
        question=(
            "Lis le fichier entier app/llm/local_llm.py."
        ),
        search_hints={"app/llm/local_llm.py": [20]},
    )
    assert arguments == {"path": "app/llm/local_llm.py"}


def test_consolidate_sources_removes_superseded_search():
    sources = [
        {
            "file": "app/llm/local_llm.py",
            "type": "tool",
            "name": "search_code",
            "content": "search result",
            "start_line": 20,
            "end_line": 20,
            "search_paths": ["app/llm/local_llm.py"],
            "search_lines": {"app/llm/local_llm.py": [20]},
        },
        {
            "file": "app/llm/local_llm.py",
            "type": "tool",
            "name": "read_file",
            "content": "file content",
            "start_line": 5,
            "end_line": 55,
        },
    ]
    result = consolidate_sources(sources)

    assert len(result) == 1
    assert result[0]["name"] == "read_file"
