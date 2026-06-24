from sentence_transformers import SentenceTransformer

def embeddings(chunk):
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # The sentences to encode
    sentences = chunk["content"]

    # 2. Calculate embeddings by calling model.encode()
    chunk["vector"] = model.encode(sentences)
    # print(embeddings.shape)
    # [3, 384]