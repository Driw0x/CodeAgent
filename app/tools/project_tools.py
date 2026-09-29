import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IGNORED_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def _resolve_project_path(path: str) -> Path:
    resolved = (PROJECT_ROOT / path).resolve()
    project_root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError("Path must stay inside the project.") from error
    return resolved


def _is_ignored(path: Path) -> bool:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    return any(part in IGNORED_DIRS for part in relative.parts)


def _project_files(
    path: Path,
    suffix: str | None = None,
) -> list[Path]:
    if path.is_file():
        if _is_ignored(path):
            return []
        if suffix is not None and path.suffix != suffix:
            return []
        return [path]
    if _is_ignored(path):
        return []
    files = []
    for root, dirs, filenames in os.walk(
        path,
        topdown=True,
        followlinks=False,
    ):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in IGNORED_DIRS
        )
        root_path = Path(root)
        for filename in sorted(filenames):
            file = root_path / filename
            if suffix is not None and file.suffix != suffix:
                continue
            files.append(file)
    return files


def read_file(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict:
    """Read a project file, optionally restricted to a line range."""
    if start_line < 1:
        raise ValueError("start_line must be greater than or equal to 1.")
    if end_line is not None and end_line < start_line:
        raise ValueError("end_line must be greater than or equal to start_line.")
    file_path = _resolve_project_path(path)
    if not file_path.is_file():
        raise ValueError(f"File does not exist: {path}")
    lines = file_path.read_text(encoding="utf-8", errors="replace",).splitlines()
    if lines and start_line > len(lines):
        raise ValueError(f"start_line exceeds file length ({len(lines)} lines).")
    actual_end = (
        len(lines)
        if end_line is None
        else min(end_line, len(lines))
    )
    content = "\n".join(lines[start_line - 1:actual_end])
    return {
        "path": file_path.relative_to(PROJECT_ROOT).as_posix(),
        "start_line": start_line,
        "end_line": actual_end,
        "content": content,
    }


def list_files(
    path: str = ".",
    max_results: int = 200,
) -> dict:
    """List project files recursively while ignoring generated directories."""
    if not 1 <= max_results <= 500:
        raise ValueError("max_results must be between 1 and 500.")
    base_path = _resolve_project_path(path)
    if not base_path.exists():
        raise ValueError(f"Path does not exist: {path}")
    files = [
        file.relative_to(PROJECT_ROOT).as_posix()
        for file in _project_files(base_path)
    ]
    return {"files": files[:max_results], "truncated": len(files) > max_results}


def search_code(
    query: str,
    path: str = ".",
    max_results: int = 50,
) -> dict:
    """Search for a literal string in Python source files."""
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty.")
    if not 1 <= max_results <= 200:
        raise ValueError("max_results must be between 1 and 200.")
    base_path = _resolve_project_path(path)
    if not base_path.exists():
        raise ValueError(f"Path does not exist: {path}")
    matches = []
    normalized_query = query.casefold()
    for file in _project_files(base_path, suffix=".py"):
        with file.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                if (
                    normalized_query
                    not in line.casefold()
                ):
                    continue
                matches.append(
                    {
                        "file": file.relative_to(PROJECT_ROOT).as_posix(),
                        "line": line_number,
                        "text": line.strip(),
                    }
                )
                if len(matches) > max_results:
                    return {
                        "matches": matches[
                            :max_results
                        ],
                        "truncated": True,
                    }
    return {"matches": matches, "truncated": False}


def get_git_diff(
    staged: bool = False,
) -> dict:
    """Return the current Git diff without modifying repository state."""
    command = [
        "git",
        "diff",
        "--no-ext-diff",
        "--no-color",
    ]
    if staged:
        command.append("--cached")
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("git diff timed out.") from error
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed.")
    return {"staged": staged, "diff": result.stdout}


def run_tests(
    path: str = "tests",
    timeout_seconds: int = 60,
) -> dict:
    """Run pytest on a project path and return the test result."""
    if not 1 <= timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be between 1 and 300.")
    test_path = _resolve_project_path(path)
    if not test_path.exists():
        raise ValueError(f"Test path does not exist: {path}")
    relative_path = test_path.relative_to(PROJECT_ROOT).as_posix()
    environment = os.environ.copy()
    environment[
        "PYTHONDONTWRITEBYTECODE"
    ] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        relative_path,
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "timed_out": True,
            "returncode": None,
            "stdout": "",
            "stderr": (
                f"pytest timed out after "
                f"{timeout_seconds} seconds."
            ),
        }
    return {
        "passed": result.returncode == 0,
        "timed_out": False,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
