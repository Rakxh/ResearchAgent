import os
import numpy as np

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

_client = None


def get_openai_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed_text(text, model=EMBEDDING_MODEL):
    client = get_openai_client()
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding, dtype="float32")


def embed_texts(texts, model=EMBEDDING_MODEL):
    client = get_openai_client()
    response = client.embeddings.create(model=model, input=texts)
    return np.array([item.embedding for item in response.data], dtype="float32")
