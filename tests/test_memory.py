import os
import numpy as np

from src.memory import ResearchMemory


def fake_embed(text):
    vector = np.zeros(8, dtype="float32")
    for i, char in enumerate(text[:8]):
        vector[i] = float(ord(char))
    return vector


def test_memory_add_and_search(tmp_path):
    index_path = os.path.join(tmp_path, "index.faiss")
    metadata_path = os.path.join(tmp_path, "metadata.json")

    memory = ResearchMemory(index_path=index_path, metadata_path=metadata_path, dim=8, embed_fn=fake_embed)
    memory.add_finding(topic="topic-a", content="abcdefgh", source="test")
    memory.add_finding(topic="topic-b", content="zzzzzzzz", source="test")

    results = memory.search("abcdefgh", k=1)
    assert len(results) == 1
    assert results[0]["topic"] == "topic-a"


def test_memory_persists_across_instances(tmp_path):
    index_path = os.path.join(tmp_path, "index.faiss")
    metadata_path = os.path.join(tmp_path, "metadata.json")

    memory = ResearchMemory(index_path=index_path, metadata_path=metadata_path, dim=8, embed_fn=fake_embed)
    memory.add_finding(topic="topic-a", content="abcdefgh", source="test")

    reloaded = ResearchMemory(index_path=index_path, metadata_path=metadata_path, dim=8, embed_fn=fake_embed)
    assert reloaded.index.ntotal == 1
    assert reloaded.metadata[0]["topic"] == "topic-a"


def test_memory_search_empty_index(tmp_path):
    index_path = os.path.join(tmp_path, "index.faiss")
    metadata_path = os.path.join(tmp_path, "metadata.json")
    memory = ResearchMemory(index_path=index_path, metadata_path=metadata_path, dim=8, embed_fn=fake_embed)
    assert memory.search("anything") == []
