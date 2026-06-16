from src.graph import route_after_review, build_graph


def make_state(is_approved, revision_count, max_revisions=2):
    return {
        "topic": "test",
        "findings": [],
        "summary": "summary",
        "report": "report",
        "human_feedback": None,
        "is_approved": is_approved,
        "revision_count": revision_count,
        "max_revisions": max_revisions,
        "final_report": None,
    }


def test_route_after_review_approved_finalizes():
    assert route_after_review(make_state(True, 1)) == "finalize"


def test_route_after_review_not_approved_revises():
    assert route_after_review(make_state(False, 1)) == "revise"


def test_route_after_review_hits_max_revisions():
    assert route_after_review(make_state(False, 2, max_revisions=2)) == "finalize"


def test_build_graph_compiles_with_expected_nodes():
    compiled = build_graph()
    node_names = set(compiled.get_graph().nodes.keys())
    assert {"research", "summarize", "write", "human_review", "finalize"}.issubset(node_names)
