import os
import json
import numpy as np
import faiss

from src.embeddings import embed_text, EMBEDDING_DIM


class ResearchMemory:
    def __init__(
        self,
        index_path="memory_store/index.faiss",
        metadata_path="memory_store/metadata.json",
        dim=EMBEDDING_DIM,
        embed_fn=None,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = dim
        self.embed_fn = embed_fn or embed_text
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []

    def save(self):
        directory = os.path.dirname(self.index_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_finding(self, topic, content, source=None):
        vector = self.embed_fn(content)
        self.index.add(np.expand_dims(vector, axis=0))
        self.metadata.append({"topic": topic, "content": content, "source": source})
        self.save()

    def search(self, query, k=5):
        if self.index.ntotal == 0:
            return []
        vector = self.embed_fn(query)
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(np.expand_dims(vector, axis=0), k)
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["distance"] = float(distance)
            results.append(entry)
        return results
