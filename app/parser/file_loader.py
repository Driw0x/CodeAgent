import os
from fnmatch import fnmatch
from pathlib import Path

from app.parser.variable import EXTENSIONS, IGNORED_DIRS, IGNORED_FILES, LANGAGE

# Lecture d'un fichier

def read(path):
    with open(path, "r", encoding="utf-8") as fichier:
        contenu = fichier.read()
    return contenu

# Lecture de tous les fichiers d'un répertoire

def read_dir(path):
    lprog = []
    pattern = f"*.{EXTENSIONS[LANGAGE.lower()]}"
    for root, directories, filenames in os.walk(path):
        directories[:] = [name for name in directories if name not in IGNORED_DIRS]
        for name in filenames:
            if name not in IGNORED_FILES and fnmatch(name, pattern):
                prog = Path(root) / name
                lprog.append({"path": prog, "content": read(prog)})
    return lprog
