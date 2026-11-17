import ast
import hashlib
from pathlib import Path

def extract_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    elif isinstance(target, ast.Tuple):
        names = []
        for elt in target.elts:
            names.extend(extract_names(elt))
        return names
    elif isinstance(target, ast.Attribute):
        return [ast.unparse(target)]
    return []
            

def chunking(data):
    ast_data = ast.parse(data["content"])
    chunk = []

    for node in ast_data.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in extract_names(target):
                    chunk.append({"file": data["path"],
                                  "type": "variable",
                                  "name": name,
                                  "content": ast.unparse(node),
                                  "start_line": node.lineno,
                                  "end_line": node.end_lineno})
        elif isinstance(node, ast.FunctionDef):
            chunk.append({"file": data["path"],
                          "type": "function",
                          "name": node.name,
                          "content": ast.unparse(node),
                          "start_line": node.lineno,
                          "end_line": node.end_lineno})
        elif isinstance(node, ast.Import):
            for lib in node.names:
                chunk.append({"file": data["path"],
                              "type": "import",
                              "name": lib.name,
                              "content": ast.unparse(node),
                              "start_line": node.lineno,
                              "end_line": node.end_lineno})
        elif isinstance(node, ast.ClassDef):
            chunk.append({"file": data["path"],
                          "type": "class",
                          "name": node.name,
                          "content": ast.unparse(node),
                          "start_line": node.lineno,
                          "end_line": node.end_lineno})
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                chunk.append({"file": data["path"],
                            "type": "import_from",
                            "module": node.module,
                            "name": name.name,
                            "content": ast.unparse(node),
                            "start_line": node.lineno,
                            "end_line": node.end_lineno})
        elif isinstance(node, ast.AnnAssign):
            for name in extract_names(node.target):
                chunk.append({"file": data["path"],
                              "type": "variable",
                              "name": name,
                              "content": ast.unparse(node),
                              "start_line": node.lineno,
                              "end_line": node.end_lineno})
        elif isinstance(node, ast.AsyncFunctionDef):
            chunk.append({"file": data["path"],
                          "type": "function",
                          "name": node.name,
                          "content": ast.unparse(node),
                          "start_line": node.lineno,
                          "end_line": node.end_lineno})

    source_version = hashlib.sha256(data["content"].encode("utf-8")).hexdigest()
    occurrences = {}
    file_key = Path(data["path"]).as_posix()

    for item in chunk:
        identity = (item["type"], item.get("module"), item["name"])
        occurrence = occurrences.get(identity, 0)
        occurrences[identity] = occurrence + 1
        stable_key = "\0".join((file_key, *(str(value or "") for value in identity), str(occurrence)))
        item["chunk_id"] = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
        item["source_version"] = source_version
        item["chunk_version"] = hashlib.sha256(item["content"].encode("utf-8")).hexdigest()

    return chunk
