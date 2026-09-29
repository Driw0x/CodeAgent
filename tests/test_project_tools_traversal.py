from pathlib import Path

import app.tools.project_tools as project_tools


def test_project_files_prunes_ignored_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    ignored_dir = tmp_path / ".venv" / "Lib" / "site-packages"
    ignored_dir.mkdir(parents=True)
    source = app_dir / "main.py"
    ignored = ignored_dir / "package.py"
    source.write_text("TARGET = True\n", encoding="utf-8")
    ignored.write_text("TARGET = True\n", encoding="utf-8")
    files = project_tools._project_files(tmp_path, suffix=".py")

    assert files == [source]


def test_search_code_ignores_generated_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    ignored_dir = tmp_path / ".venv" / "Lib" / "site-packages"
    ignored_dir.mkdir(parents=True)
    (app_dir / "main.py").write_text("class LocalLLM:\n" "    pass\n", encoding="utf-8")
    (ignored_dir / "duplicate.py").write_text("class LocalLLM:\n" "    pass\n", encoding="utf-8")
    result = project_tools.search_code(query="LocalLLM")

    assert result == {
        "matches": [
            {
                "file": "app/main.py",
                "line": 1,
                "text": "class LocalLLM:",
            }
        ],
        "truncated": False,
    }


def test_search_code_only_reads_python_files(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "source.py").write_text("LocalLLM = True\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("LocalLLM\n", encoding="utf-8")
    result = project_tools.search_code(query="LocalLLM")
    assert len(result["matches"]) == 1
    assert (result["matches"][0]["file"] == "source.py")


def test_project_files_are_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)

    for path in (Path("z.py"), Path("a.py"), Path("folder/b.py")):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("", encoding="utf-8")

    files = project_tools._project_files(tmp_path, suffix=".py")
    relative = [file.relative_to(tmp_path).as_posix() for file in files]

    assert relative == ["a.py", "z.py", "folder/b.py"]
