from app.memory.embeddings import chunk_embedding_text


def test_chunk_embedding_text_contains_metadata():
    chunk = {
        "file": "app/parser/file_loader.py",
        "type": "function",
        "name": "read_dir",
        "content": "def read_dir(path): ...",
    }

    result = chunk_embedding_text(chunk)

    assert "app/parser/file_loader.py" in result
    assert "function" in result
    assert "read_dir" in result
    assert "def read_dir(path)" in result