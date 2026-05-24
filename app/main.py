from app.parser.file_loader import *

p = r"e:\workspace\git_workspace\CodeAgent"

for i in read_dir(p):
    print(i["path"].name)