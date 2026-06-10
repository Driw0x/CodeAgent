import ast

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

    return chunk