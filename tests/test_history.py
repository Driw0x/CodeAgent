import json

from app.memory import save_analysis


def test_save_analysis_creates_history(tmp_path):
    save_analysis(
        tmp_path,
        "Question 1",
        "Réponse 1",
    )

    history_file = tmp_path / "history.jsonl"

    assert history_file.exists()

    entry = json.loads(
        history_file.read_text(encoding="utf-8").strip()
    )

    assert entry["question"] == "Question 1"
    assert entry["answer"] == "Réponse 1"
    assert "timestamp" in entry


def test_save_analysis_appends_history(tmp_path):
    save_analysis(
        tmp_path,
        "Question 1",
        "Réponse 1",
    )

    save_analysis(
        tmp_path,
        "Question 2",
        "Réponse 2",
    )

    lines = (
        tmp_path / "history.jsonl"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["question"] == "Question 1"
    assert second["question"] == "Question 2"