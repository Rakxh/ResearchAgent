from src.memory import ResearchMemory
from src.knowledge_base import load_index, search_index, DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH

_memory = None
_knowledge_base = None


def get_memory():
    global _memory
    if _memory is None:
        _memory = ResearchMemory()
    return _memory


def get_knowledge_base(index_path=DEFAULT_INDEX_PATH, metadata_path=DEFAULT_METADATA_PATH):
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = load_index(index_path, metadata_path)
    return _knowledge_base


def search_knowledge_base(query: str, max_results: int = 4) -> str:
    """Search the curated research knowledge base for passages relevant to a query."""
    index, metadata = get_knowledge_base()
    results = search_index(query, index, metadata, k=max_results)
    if not results:
        return "No relevant passages found."
    return "\n\n".join(f"[{r['source']}] {r['text']}" for r in results)


def record_finding(topic: str, content: str, source: str = "") -> str:
    """Save a research finding to persistent memory so it can be recalled in future sessions."""
    get_memory().add_finding(topic=topic, content=content, source=source or None)
    return "Finding recorded to persistent memory."


def recall_findings(query: str, max_results: int = 5) -> str:
    """Recall the most relevant past research findings from persistent memory for a query."""
    results = get_memory().search(query, k=max_results)
    if not results:
        return "No relevant past findings."
    return "\n".join(f"[{r['topic']}] {r['content']}" for r in results)
