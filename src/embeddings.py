import numpy as np

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def embed_text(text):
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return np.array(vector, dtype="float32")


def embed_texts(texts):
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=16)
    return np.array(vectors, dtype="float32")
