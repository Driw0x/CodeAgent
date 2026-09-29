import subprocess
import sys
from unittest.mock import patch

import pytest

from app.tools import project_tools


def test_read_file(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    file = tmp_path / "example.py"
    file.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
    result = project_tools.read_file("example.py", start_line=2, end_line=3)
    assert result["path"] == "example.py"
    assert result["start_line"] == 2
    assert result["end_line"] == 3
    assert result["content"] == "line 2\nline 3"


def test_read_file_rejects_path_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the project"):
        project_tools.read_file("../outside.txt")


def test_list_files(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_main.py").write_text("", encoding="utf-8")
    result = project_tools.list_files()
    assert "app/main.py" in result["files"]
    assert "tests/test_main.py" in result["files"]
    assert result["truncated"] is False


def test_list_files_ignores_generated_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "module.py").write_text("", encoding="utf-8")
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    result = project_tools.list_files()
    assert "main.py" in result["files"]
    assert ".venv/module.py" not in result["files"]


def test_search_code(tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    file = tmp_path / "module.py"
    file.write_text("def hello():\n" "    return 'world'\n", encoding="utf-8")
    result = project_tools.search_code("hello")
    assert result["matches"] == [{"file": "module.py", "line": 1, "text": "def hello():"}]


@patch("app.tools.project_tools.subprocess.run")
def test_get_git_diff(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="diff --git a/a.py b/a.py",
        stderr="",
    )
    result = project_tools.get_git_diff()
    assert result["staged"] is False
    assert "diff --git" in result["diff"]
    command = mock_run.call_args.args[0]
    assert command[:2] == ["git", "diff"]


@patch("app.tools.project_tools.subprocess.run")
def test_run_tests(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="5 passed",
        stderr="",
    )
    result = project_tools.run_tests()
    assert result["passed"] is True
    assert result["timed_out"] is False
    assert result["returncode"] == 0
    assert result["stdout"] == "5 passed"
    command = mock_run.call_args.args[0]
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "pytest"]
    assert "tests" in command
