from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from src.state import ResearchState
from src.agents import build_researcher_agent, build_summarizer_agent, build_writer_agent, run_agent_task


async def research_node(state):
    agent = build_researcher_agent()
    task = (
        f"Research the topic '{state['topic']}'. Use recall_findings first to check for relevant prior "
        "research, then use search_knowledge_base to gather information from the research knowledge "
        "base, and use record_finding to save the most useful new findings. Reply with a structured "
        "list of findings, each with a short claim and source."
    )
    response = await run_agent_task(agent, task)
    new_finding = {"query": state["topic"], "content": response, "source": None}
    return {"findings": state["findings"] + [new_finding]}


async def summarize_node(state):
    agent = build_summarizer_agent()
    findings_text = "\n\n".join(finding["content"] for finding in state["findings"])
    task = f"Summarize these research findings on '{state['topic']}':\n\n{findings_text}"
    summary = await run_agent_task(agent, task)
    return {"summary": summary}


async def write_node(state):
    agent = build_writer_agent()
    feedback_text = state.get("human_feedback") or "None. This is the first draft."
    task = (
        f"Write a structured research report on '{state['topic']}' based on this summary:\n\n"
        f"{state['summary']}\n\nHuman reviewer feedback to address, if any: {feedback_text}"
    )
    report = await run_agent_task(agent, task)
    return {"report": report, "revision_count": state["revision_count"] + 1}


def human_review_node(state):
    decision = interrupt(
        {
            "type": "human_review",
            "topic": state["topic"],
            "report": state["report"],
            "instructions": "Reply 'approve' to accept the report, or provide feedback text to request a revision.",
        }
    )
    decision_text = str(decision).strip()
    if decision_text.lower() in {"approve", "approved", "yes", "y"}:
        return {"is_approved": True, "human_feedback": None}
    return {"is_approved": False, "human_feedback": decision_text}


def finalize_node(state):
    return {"final_report": state["report"]}


def route_after_review(state):
    if state["is_approved"] or state["revision_count"] >= state["max_revisions"]:
        return "finalize"
    return "revise"


def build_graph(checkpointer=None):
    graph = StateGraph(ResearchState)
    graph.add_node("research", research_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("write", write_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", "write")
    graph.add_edge("write", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {"revise": "write", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    checkpointer = checkpointer or MemorySaver()
    return graph.compile(checkpointer=checkpointer)
