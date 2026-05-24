from pathlib import Path
from app.parser.variable import *

# Lecture d'un fichier

def read(path):
    with open(path, "r", encoding="utf-8") as fichier:
        contenu = fichier.read()
    return contenu

# Lecture de tous les fichiers d'un répertoire

def read_dir(path):
    p = Path(path)
    lprog = []
    for prog in p.glob('**/*.%s'%EXTENSIONS[LANGAGE.lower()]):
        if not IGNORED_DIRS.intersection(prog.parts):
            lprog.append({"path": prog, "content": read(prog)})
    return lprog