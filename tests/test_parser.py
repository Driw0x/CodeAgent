from textwrap import dedent

from app.parser.chunker import chunking
from app.parser.file_loader import read, read_dir


def code(text):
    return dedent(text).strip()


# =========================
# Tests chunker.py
# =========================

def test_variable_simple():
    data = {"path": "test.py",
            "content": "x = 42"}

    chunks = chunking(data)

    assert len(chunks) == 1
    assert chunks[0]["file"] == "test.py"
    assert chunks[0]["type"] == "variable"
    assert chunks[0]["name"] == "x"
    assert chunks[0]["content"] == "x = 42"
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 1


def test_plusieurs_variables_separees():
    data = {"path": "test.py",
            "content": code("""
                            x = 1
                            y = 2
                            z = 3
                            """)}

    chunks = chunking(data)

    assert len(chunks) == 3
    assert [chunk["name"] for chunk in chunks] == ["x", "y", "z"]
    assert all(chunk["type"] == "variable" for chunk in chunks)


def test_assignation_multiple_meme_ligne():
    data = {"path": "test.py",
            "content": "x = y = 0"}

    chunks = chunking(data)

    assert len(chunks) == 2
    assert [chunk["name"] for chunk in chunks] == ["x", "y"]
    assert all(chunk["content"] == "x = y = 0" for chunk in chunks)


def test_tuple_assignment():
    data = {"path": "test.py",
            "content": "a, b = 1, 2"}

    chunks = chunking(data)

    assert len(chunks) == 2
    assert [chunk["name"] for chunk in chunks] == ["a", "b"]
    assert all(chunk["type"] == "variable" for chunk in chunks)


def test_annotation_variable():
    data = {"path": "test.py",
            "content": "age: int = 20"}

    chunks = chunking(data)

    assert len(chunks) == 1
    assert chunks[0]["type"] == "variable"
    assert chunks[0]["name"] == "age"
    assert chunks[0]["content"] == "age: int = 20"


def test_function():
    data = {"path": "test.py",
            "content": code("""
                            def hello():
                                return "world"
                        """)}

    chunks = chunking(data)

    assert len(chunks) == 1
    assert chunks[0]["type"] == "function"
    assert chunks[0]["name"] == "hello"
    assert "def hello" in chunks[0]["content"]


def test_async_function():
    data = {"path": "test.py",
            "content": code("""
                            async def fetch_data():
                                return 1
                            """)}

    chunks = chunking(data)

    assert len(chunks) == 1
    assert chunks[0]["type"] == "function"
    assert chunks[0]["name"] == "fetch_data"
    assert "async def fetch_data" in chunks[0]["content"]


def test_class():
    data = {"path": "test.py",
            "content": code("""
                            class User:
                                pass
                            """)}

    chunks = chunking(data)

    assert len(chunks) == 1
    assert chunks[0]["type"] == "class"
    assert chunks[0]["name"] == "User"
    assert "class User" in chunks[0]["content"]


def test_import_simple():
    data = {"path": "test.py",
            "content": "import os"}

    chunks = chunking(data)

    assert len(chunks) == 1
    assert chunks[0]["type"] == "import"
    assert chunks[0]["name"] == "os"
    assert chunks[0]["content"] == "import os"


def test_import_multiple():
    data = {"path": "test.py",
            "content": "import os, sys"}

    chunks = chunking(data)

    assert len(chunks) == 2
    assert [chunk["name"] for chunk in chunks] == ["os", "sys"]
    assert all(chunk["type"] == "import" for chunk in chunks)


def test_import_from():
    data = {"path": "test.py",
            "content": "from pathlib import Path"}

    chunks = chunking(data)

    assert len(chunks) == 1
    assert chunks[0]["type"] == "import_from"
    assert chunks[0]["module"] == "pathlib"
    assert chunks[0]["name"] == "Path"
    assert chunks[0]["content"] == "from pathlib import Path"


def test_import_from_multiple():
    data = {"path": "test.py",
            "content": "from os import path, mkdir"}

    chunks = chunking(data)

    assert len(chunks) == 2
    assert [chunk["type"] for chunk in chunks] == ["import_from", "import_from"]
    assert [chunk["module"] for chunk in chunks] == ["os", "os"]
    assert [chunk["name"] for chunk in chunks] == ["path", "mkdir"]


def test_fichier_complet():
    data = {"path": "test.py",
            "content": code("""
                            import os
                            from pathlib import Path

                            x = 10
                            name: str = "Victor"

                            def add(a, b):
                                return a + b

                            class User:
                                pass

                            async def main():
                                return None
                            """)}

    chunks = chunking(data)

    assert len(chunks) == 7

    assert [chunk["type"] for chunk in chunks] == [
        "import",
        "import_from",
        "variable",
        "variable",
        "function",
        "class",
        "function",
    ]

    assert [chunk["name"] for chunk in chunks] == [
        "os",
        "Path",
        "x",
        "name",
        "add",
        "User",
        "main",
    ]


def test_fichier_vide():
    data = {"path": "test.py",
            "content": ""}

    chunks = chunking(data)

    assert chunks == []


def test_code_non_chunkable():
    data = {"path": "test.py",
            "content": code("""
                            print("hello")
                            if True:
                                x = 1
                            """)}

    chunks = chunking(data)

    assert chunks == []


# =========================
# Tests file_loader.py
# =========================

def test_read_lit_le_contenu_du_fichier(tmp_path):
    fichier = tmp_path / "exemple.py"
    fichier.write_text("x = 42", encoding="utf-8")

    contenu = read(fichier)

    assert contenu == "x = 42"


def test_read_dir_lit_les_fichiers_python(tmp_path):
    fichier1 = tmp_path / "a.py"
    fichier2 = tmp_path / "b.py"

    fichier1.write_text("x = 1", encoding="utf-8")
    fichier2.write_text("y = 2", encoding="utf-8")

    resultats = read_dir(tmp_path)

    assert len(resultats) == 2

    paths = [resultat["path"].name for resultat in resultats]
    contents = [resultat["content"] for resultat in resultats]

    assert "a.py" in paths
    assert "b.py" in paths
    assert "x = 1" in contents
    assert "y = 2" in contents


def test_read_dir_ignore_les_fichiers_non_python(tmp_path):
    fichier_python = tmp_path / "main.py"
    fichier_txt = tmp_path / "notes.txt"

    fichier_python.write_text("x = 1", encoding="utf-8")
    fichier_txt.write_text("ceci ne doit pas être lu", encoding="utf-8")

    resultats = read_dir(tmp_path)

    assert len(resultats) == 1
    assert resultats[0]["path"].name == "main.py"
    assert resultats[0]["content"] == "x = 1"


def test_read_dir_ignore_les_dossiers_ignores(tmp_path):
    dossier_cache = tmp_path / "__pycache__"
    dossier_cache.mkdir()

    fichier_ignore = dossier_cache / "cache.py"
    fichier_valide = tmp_path / "main.py"

    fichier_ignore.write_text("x = 1", encoding="utf-8")
    fichier_valide.write_text("y = 2", encoding="utf-8")

    resultats = read_dir(tmp_path)

    assert len(resultats) == 1
    assert resultats[0]["path"].name == "main.py"
    assert resultats[0]["content"] == "y = 2"


def test_read_dir_lit_les_fichiers_dans_sous_dossiers(tmp_path):
    sous_dossier = tmp_path / "src"
    sous_dossier.mkdir()

    fichier = sous_dossier / "module.py"
    fichier.write_text("def hello():\n    pass", encoding="utf-8")

    resultats = read_dir(tmp_path)

    assert len(resultats) == 1
    assert resultats[0]["path"].name == "module.py"
    assert "def hello" in resultats[0]["content"]