from app.parser.file_loader import *
from app.parser.chunker import *
from app.memory.embeddings import *
from app.memory.vector_store import *

p = Path(r"e:\workspace\git_workspace\CodeAgent")
chunks = []

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.IndexFlatL2(384)   # build the index
# print(index.is_trained)

for i in read_dir(p):
    # print(i["path"].relative_to(p.parent))
    for elt in chunking(i):
        v = embeddings(model, elt["content"])
        store_vector(index, v.reshape(1, -1))
        chunks.append(elt)
        # print(elt)
    # print()

k = 3
req = "function that reads file content"
v = embeddings(model, req).reshape(1, -1)
D, neighbors = k_neighbors(index, chunks, v, k)
print(D, neighbors)
print(req)
for i in range(k):
    print(i+1, ":", end=" ")
    print(neighbors[i]["file"])
    print(neighbors[i]["name"])
    print(f"score : {D[i]}")
    print()

req = "function that reads directory"
v = embeddings(model, req).reshape(1, -1)
D, neighbors = k_neighbors(index, chunks, v, k)
print(req)
for i in range(k):
    print(i+1, ":", end=" ")
    print(neighbors[i]["file"])
    print(neighbors[i]["name"])
    print(f"score : {D[i]}")
    print()