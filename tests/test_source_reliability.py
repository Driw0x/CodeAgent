from app.rag.reliability import prepare_reliable_sources


def test_stale_sources_are_removed_with_current_version_map():
    chunks = [
        {"file": "a.py", "source_version": "old", "content": "old"},
        {"file": "a.py", "source_version": "new", "content": "new"},
    ]

    current, contradictions = prepare_reliable_sources(chunks, {"a.py": "new"})

    assert [chunk["content"] for chunk in current] == ["new"]
    assert contradictions == set()


def test_explicit_source_group_contradiction_is_marked():
    chunks = [
        {"file": "a.py", "source_group": "setting", "chunk_version": "one", "content": "enabled"},
        {"file": "b.py", "source_group": "setting", "chunk_version": "two", "content": "disabled"},
        {"file": "c.py", "source_group": "other", "chunk_version": "same", "content": "compatible"},
    ]

    current, contradictions = prepare_reliable_sources(chunks)

    assert contradictions == {"setting"}
    assert [chunk.get("reliability") for chunk in current] == ["contradictory", "contradictory", None]
