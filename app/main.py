from app.parser.file_loader import *
from app.parser.chunker import *

p = Path("e:\workspace\git_workspace\CodeAgent")

for i in read_dir(p):
    print(i["path"].relative_to(p.parent))
    for elt in chunking(i):
        print(elt)
    print()