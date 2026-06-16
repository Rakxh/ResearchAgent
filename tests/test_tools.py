from unittest.mock import patch
import src.tools as tools_module


def test_search_knowledge_base_formats_results():
    fake_results = [
        {"source": "doc_a.txt", "text": "Some passage about doc a.", "score": 0.9},
        {"source": "doc_b.txt", "text": "Some passage about doc b.", "score": 0.8},
    ]

    with patch.object(tools_module, "get_knowledge_base", return_value=(None, None)), \
         patch.object(tools_module, "search_index", return_value=fake_results):
        output = tools_module.search_knowledge_base("anything")

    assert "[doc_a.txt]" in output
    assert "Some passage about doc a." in output
    assert "[doc_b.txt]" in output


def test_search_knowledge_base_no_results():
    with patch.object(tools_module, "get_knowledge_base", return_value=(None, None)), \
         patch.object(tools_module, "search_index", return_value=[]):
        output = tools_module.search_knowledge_base("anything")

    assert output == "No relevant passages found."


def test_record_and_recall_findings():
    class FakeMemory:
        def __init__(self):
            self.added = []

        def add_finding(self, topic, content, source=None):
            self.added.append((topic, content, source))

        def search(self, query, k=5):
            return [{"topic": "topic-a", "content": "fact about topic a"}]

    fake_memory = FakeMemory()

    with patch.object(tools_module, "get_memory", return_value=fake_memory):
        result = tools_module.record_finding("topic-a", "fact about topic a", source="doc_a.txt")
        assert result == "Finding recorded to persistent memory."
        assert fake_memory.added == [("topic-a", "fact about topic a", "doc_a.txt")]

        recall_output = tools_module.recall_findings("topic-a")
        assert "[topic-a] fact about topic a" in recall_output


def test_recall_findings_no_results():
    class FakeMemory:
        def search(self, query, k=5):
            return []

    with patch.object(tools_module, "get_memory", return_value=FakeMemory()):
        output = tools_module.recall_findings("anything")

    assert output == "No relevant past findings."
