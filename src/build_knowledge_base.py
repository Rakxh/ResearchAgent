from src.knowledge_base import build_index

if __name__ == "__main__":
    index, metadata = build_index()
    print(f"Indexed {len(metadata)} chunks from data/knowledge_base into vector_db/")
