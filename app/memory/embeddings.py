from sentence_transformers import SentenceTransformer

def embeddings(model, data):
    # The sentences to encode
    sentences = data

    # 2. Calculate embeddings by calling model.encode()
    return model.encode(sentences, show_progress_bar=False)
    # print(embeddings.shape)
    # [3, 384]