import faiss
import numpy as np
import pytest
from pathlib import Path

from app.memory import index_exists, load_index, save_index
from app.utils.paths import project_memory_dir


def test_save_and_load_index(tmp_path):
    index = faiss.IndexFlatL2(2)
    index.add(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype="float32",
        )
    )

    chunks = [
        {"file": Path("a.py")},
        {"file": Path("b.py")},
    ]

    save_index(index, chunks, tmp_path)

    assert index_exists(tmp_path)

    loaded_index, loaded_chunks = load_index(tmp_path)

    assert loaded_index.ntotal == 2
    assert loaded_chunks == [
        {"file": "a.py"},
        {"file": "b.py"},
    ]

    query = np.array([[1.0, 2.0]], dtype="float32")

    _, expected_indices = index.search(query, 2)
    _, loaded_indices = loaded_index.search(query, 2)

    np.testing.assert_array_equal(
        loaded_indices,
        expected_indices,
    )


def test_load_index_rejects_mismatched_chunks(tmp_path):
    index = faiss.IndexFlatL2(2)
    index.add(
        np.array(
            [[1.0, 2.0]],
            dtype="float32",
        )
    )

    save_index(
        index,
        [{"file": "a.py"}],
        tmp_path,
    )

    (tmp_path / "chunks.json").write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Index/chunks mismatch",
    ):
        load_index(tmp_path)


def test_project_memory_dir_separates_projects_with_same_name(tmp_path):
    first_project = tmp_path / "first" / "backend"
    second_project = tmp_path / "second" / "backend"

    first_project.mkdir(parents=True)
    second_project.mkdir(parents=True)

    memory_root = tmp_path / "memory"

    first_memory = project_memory_dir(first_project, memory_root)
    second_memory = project_memory_dir(second_project, memory_root)

    assert first_memory != second_memory
    assert first_memory.parent == memory_root
    assert second_memory.parent == memory_root