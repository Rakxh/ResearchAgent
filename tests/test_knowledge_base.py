import os
import numpy as np

from src.knowledge_base import build_index, load_index, search_index, chunk_text


def fake_embed_texts(texts):
    vectors = []
    for text in texts:
        vector = np.zeros(8, dtype="float32")
        for i, char in enumerate(text[:8]):
            vector[i] = float(ord(char))
        vectors.append(vector)
    return np.array(vectors, dtype="float32")


def write_sample_docs(directory):
    with open(os.path.join(directory, "topic_a.txt"), "w", encoding="utf-8") as f:
        f.write("abcdefgh is about topic a and nothing else at all.")
    with open(os.path.join(directory, "topic_b.txt"), "w", encoding="utf-8") as f:
        f.write("zzzzzzzz is about topic b and nothing else at all.")


def test_chunk_text_respects_size():
    text = "a" * 1500
    chunks = chunk_text(text, chunk_size=600, overlap=80)
    assert len(chunks) > 1
    assert all(len(chunk) <= 600 for chunk in chunks)


def test_build_and_search_index(tmp_path):
    source_dir = os.path.join(tmp_path, "docs")
    os.makedirs(source_dir)
    write_sample_docs(source_dir)

    index_path = os.path.join(tmp_path, "index.faiss")
    metadata_path = os.path.join(tmp_path, "metadata.json")

    index, metadata = build_index(
        source_dir=source_dir,
        index_path=index_path,
        metadata_path=metadata_path,
        embed_fn=fake_embed_texts,
    )

    assert index.ntotal == len(metadata)

    results = search_index("abcdefgh", index, metadata, k=1, embed_fn=fake_embed_texts)
    assert len(results) == 1
    assert results[0]["source"] == "topic_a.txt"


def test_load_index_round_trip(tmp_path):
    source_dir = os.path.join(tmp_path, "docs")
    os.makedirs(source_dir)
    write_sample_docs(source_dir)

    index_path = os.path.join(tmp_path, "index.faiss")
    metadata_path = os.path.join(tmp_path, "metadata.json")

    build_index(
        source_dir=source_dir,
        index_path=index_path,
        metadata_path=metadata_path,
        embed_fn=fake_embed_texts,
    )

    reloaded_index, reloaded_metadata = load_index(index_path, metadata_path)
    assert reloaded_index.ntotal == len(reloaded_metadata)
    assert len(reloaded_metadata) > 0
