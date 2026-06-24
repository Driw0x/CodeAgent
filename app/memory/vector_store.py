import faiss                   # make faiss available

def store_vector(index, v):
    index.add(v)                  # add vectors to the index
    # print(index.ntotal)

def k_neighbors(index, chunks, v, k):
    D, I = index.search(v, k)
    return D[0], [chunks[i] for i in I[0]]