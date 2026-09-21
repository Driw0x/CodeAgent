import json
from datetime import datetime, timezone
from pathlib import Path


HISTORY_FILE = "history.jsonl"


def save_analysis(
    directory: str | Path,
    question: str,
    answer: str,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
    }

    with (directory / HISTORY_FILE).open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(entry, ensure_ascii=False) + "\n"
        )