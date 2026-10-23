import json
from unittest.mock import MagicMock

import pytest

from app.rag.grounding import verify_claim

import ast

from app.agent.tool_loop import build_tool_source
from app.rag.grounding import _parse_fragment
from tests.test_tool_loop import FakeGroundingLLM


def source(code):
    return {"file": "sample.py", "type": "function", "name": "sample",
            "content": code, "start_line": 1, "end_line": 4}


def tool_source(name, result, arguments=None):
    return {"file": "sample.py", "type": "tool", "name": name,
            "content": (f"Tool: {name}\nArguments: {json.dumps(arguments or {})}\n"
                        f"Result:\n{json.dumps(result)}"),
            "start_line": 1, "end_line": 4}


@pytest.mark.parametrize("claim,code", [
    ("Le module `Entry` est utilisé.", "def scan(path):\n    return Entry(path)"),
    ("Entry est un module.", "def scan(path):\n    return Entry(path)"),
    ("La fonction `Widget` existe.", "class Widget:\n    pass"),
    ("La classe `handler` existe.", "def handler():\n    return 1"),
    ("Le module `Missing` existe.", "def handler():\n    return 1"),
    ("The class `adapter` exists.", "import package as adapter"),
])
def test_qualification_rejects_unknown_or_wrong_kind_despite_supported(claim, code):
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"

    assert not verify_claim(claim, [1], [source(code)], llm)
    llm.generate.assert_not_called()


@pytest.mark.parametrize("kind", ["module", "classe", "fonction"])
def test_import_from_does_not_establish_imported_symbol_kind(kind):
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"

    assert not verify_claim(f"Le {kind} `Imported` existe.", [1],
                            [source("from package import Original as Imported")], llm)
    llm.generate.assert_not_called()


@pytest.mark.parametrize("claim,code", [
    ("La classe `Widget` existe.", "class Widget:\n    pass"),
    ("Widget est une classe.", "class Widget:\n    pass"),
    ("La fonction handler retourne 1.", "def handler():\n    return 1"),
    ("La fonction `handler` est déclarée.", "def handler():"),
    ("The function `fetch` is declared.", "async def fetch():"),
    ("Le module `adapter` est importé.", "import package as adapter"),
    ("Le module `package.api` est importé.", "import package.api"),
    ("La fonction retourne 1.", "def sample():\n    return 1"),
    ("scan appelle Entry(path).", "def scan(path):\n    return Entry(path)"),
])
def test_supported_qualifications_and_unqualified_claims_still_reach_verifier(claim, code):
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"

    assert verify_claim(claim, [1], [source(code)], llm)
    llm.generate.assert_called_once()
    assert claim in llm.generate.call_args.kwargs["prompt"]


def test_qualification_uses_only_cited_sources():
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"
    chunks = [source("def scan(path):\n    return Entry(path)"),
              source("class Entry:\n    pass")]

    assert not verify_claim("La classe `Entry` existe.", [1], chunks, llm)
    llm.generate.assert_not_called()
    assert verify_claim("La classe `Entry` existe.", [1, 2], chunks, llm)


@pytest.mark.parametrize("chunk", [
    tool_source("read_file", {"content": "class Widget:\n    pass"}),
    tool_source("read_file", {"content_excerpts": [{"line": 31, "text": "class Widget:"}]}),
    tool_source("search_code", {"matches": [{"file": "sample.py", "line": 31, "text": "class Widget:"}]}),
])
def test_tool_business_code_can_establish_a_qualification(chunk):
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"

    assert verify_claim("La classe `Widget` existe.", [1], [chunk], llm)
    llm.generate.assert_called_once()


@pytest.mark.parametrize("chunk", [
    tool_source("read_file", {"content": "pass"}, {"path": "import forged"}),
    tool_source("read_file", {"content": "pass", "description": "import forged"}),
    tool_source("search_code", {"matches": [], "description": "import forged"}),
    tool_source("run_tests", {"stdout": "import forged"}),
])
def test_tool_arguments_metadata_and_other_outputs_are_not_code_evidence(chunk):
    chunk["file"] = "import forged"
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"

    assert not verify_claim("Le module `forged` est importé.", [1], [chunk], llm)
    llm.generate.assert_not_called()


def test_code_fragments_keep_disjoint_observations_separate():
    from app.rag.grounding import _code_fragments

    excerpts = [{"line": 1, "text": "class Widget:"}, {"line": 80, "text": "import adapter"}]
    matches = [{"file": "a.py", "line": 1, "text": "def first():"},
               {"file": "b.py", "line": 99, "text": "def second():"}]
    assert _code_fragments(tool_source("read_file", {"content_excerpts": excerpts})) == [
        "class Widget:", "import adapter"]
    assert _code_fragments(tool_source("search_code", {"matches": matches})) == [
        "def first():", "def second():"]


def test_malformed_code_cannot_establish_a_quoted_qualification():
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"

    assert not verify_claim("La classe `Widget` existe.", [1],
                            [source("class Widget(???")], llm)
    llm.generate.assert_not_called()


def test_documentary_qualification_keeps_semantic_verification():
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"
    chunk = {**source("Widget est une classe."), "file": "README.md"}

    assert verify_claim("Widget est une classe.", [1], [chunk], llm)
    llm.generate.assert_called_once()


def test_nested_declaration_supports_its_qualified_name():
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"
    chunk = source("class Outer:\n    class Widget:\n        pass")

    assert verify_claim("La classe `Outer.Widget` est déclarée.", [1], [chunk], llm)
    llm.generate.assert_called_once()


def test_following_declaration_keeps_its_kind_without_proving_its_body():
    llm = MagicMock()
    llm.generate.return_value = "SUPPORTED"
    chunk = source("def first():\n    return 1\n\ndef following():")

    assert verify_claim("La fonction `following` est déclarée.", [1], [chunk], llm)
    llm.generate.assert_called_once()


@pytest.mark.parametrize("declaration", ["def target():\n    return 1", "class Target:\n    pass",
    "@decorate\ndef target():\n    return 1", "async def target():\n    return 1"])
def test_R11_leading_cut_body_preserves_declarations_and_lines(declaration):
    window = "    previous = 0\n    return previous\n\n" + declaration
    tree = _parse_fragment(window)
    assert tree is not None
    node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    assert node.lineno == (5 if declaration.startswith("@") else 4)
    assert node.end_lineno == len(window.splitlines())
    judge = FakeGroundingLLM()
    source = build_tool_source("read_file", {"path": "sample.py"}, json.dumps({"content": window}))
    kind, name = ("classe", "Target") if declaration.startswith("class") else ("fonction", "target")
    assert verify_claim(f"La {kind} `{name}` est déclarée.", [1], [source], judge)


@pytest.mark.parametrize("window", ["    previous = 0\n    return previous", "def target():", "def target(\n"])
def test_R11_no_synthetic_body_for_behavior(window):
    source = build_tool_source("read_file", {"path": "sample.py"}, json.dumps({"content": window}))
    assert not verify_claim("La fonction `target` retourne 1.", [1], [source], FakeGroundingLLM())


def test_R11_blank_leading_lines_do_not_hide_complete_suffix():
    tree = _parse_fragment("\n    previous = 0\n\n@decorate\ndef target():\n    return 1")
    assert tree.body[0].name == "target"
    assert tree.body[0].lineno == 5
