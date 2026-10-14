import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = "history.jsonl"


def describe_error(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "\n".join(describe_error(child) for child in error.exceptions)
    return f"{type(error).__name__}: {error}"


def save_analysis(directory: str | Path, question: str, answer: str, *, session_id=None, observations=None, diagnostics=None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
    }
    if session_id is not None:
        entry["session_id"] = session_id
        entry["observations"] = observations or []
    if diagnostics:
        entry["diagnostics"] = diagnostics

    history_path = directory / HISTORY_FILE
    # Preserve a partial last record; separate the next valid JSONL entry.
    separator = ""
    if history_path.exists() and history_path.stat().st_size:
        with history_path.open("rb") as file:
            file.seek(-1, 2)
            if file.read(1) != b"\n":
                separator = "\n"
    with history_path.open("a", encoding="utf-8") as file:
        file.write(separator)
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
