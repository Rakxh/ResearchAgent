import os
import json
import faiss

from src.embeddings import embed_texts

DEFAULT_INDEX_PATH = "vector_db/knowledge_base.faiss"
DEFAULT_METADATA_PATH = "vector_db/knowledge_base_metadata.json"


def load_documents(source_dir="data/knowledge_base"):
    documents = []
    if not os.path.isdir(source_dir):
        return documents
    for filename in sorted(os.listdir(source_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(source_dir, filename), "r", encoding="utf-8") as f:
                documents.append({"source": filename, "text": f.read()})
    return documents


def chunk_text(text, chunk_size=600, overlap=80):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap

    return chunks


def build_index(
    source_dir="data/knowledge_base",
    index_path=DEFAULT_INDEX_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
    embed_fn=None,
):
    embed_fn = embed_fn or embed_texts
    documents = load_documents(source_dir)

    chunks = []
    metadata = []
    for document in documents:
        for chunk in chunk_text(document["text"]):
            chunks.append(chunk)
            metadata.append({"source": document["source"], "text": chunk})

    if not chunks:
        raise ValueError(f"No documents found in {source_dir}")

    embeddings = embed_fn(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    directory = os.path.dirname(index_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return index, metadata


def load_index(index_path=DEFAULT_INDEX_PATH, metadata_path=DEFAULT_METADATA_PATH):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search_index(query, index, metadata, k=4, embed_fn=None):
    embed_fn = embed_fn or embed_texts
    k = min(k, index.ntotal)
    if k == 0:
        return []

    query_embedding = embed_fn([query])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**metadata[idx], "score": float(score)})
    return results
