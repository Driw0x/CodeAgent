from pathlib import Path

from app.memory import (
    build_manifest,
    compare_manifests,
    load_manifest,
    save_manifest,
)


def test_build_manifest_is_stable(tmp_path):
    files = [
        {
            "path": tmp_path / "a.py",
            "content": "print('hello')",
        },
    ]

    first = build_manifest(files, tmp_path)
    second = build_manifest(files, tmp_path)

    assert first == second
    assert "a.py" in first


def test_compare_manifests_detects_changes():
    previous = {
        "unchanged.py": "aaa",
        "modified.py": "bbb",
        "deleted.py": "ccc",
    }

    current = {
        "unchanged.py": "aaa",
        "modified.py": "changed",
        "added.py": "ddd",
    }

    changes = compare_manifests(
        previous,
        current,
    )

    assert changes["added"] == ["added.py"]
    assert changes["modified"] == ["modified.py"]
    assert changes["deleted"] == ["deleted.py"]
    assert changes["unchanged"] == ["unchanged.py"]


def test_save_and_load_manifest(tmp_path):
    manifest = {
        "app/main.py": "abc",
        "app/test.py": "def",
    }

    save_manifest(
        manifest,
        tmp_path,
    )

    loaded = load_manifest(tmp_path)

    assert loaded == manifest