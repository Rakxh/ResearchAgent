import os
import sys
import asyncio
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph import build_graph
from src.knowledge_base import build_index, DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH
from src.generate_knowledge_base import generate_knowledge_base

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 Multi-Agent Research Assistant")
st.caption("Researcher → Summarizer → Writer agents with human-in-the-loop review · Powered by Groq + LangGraph")


# ── helpers ──────────────────────────────────────────────────────────────────

def knowledge_base_exists():
    return (
        os.path.isfile(DEFAULT_INDEX_PATH)
        and os.path.isfile(DEFAULT_METADATA_PATH)
    )


def build_knowledge_base_if_needed():
    if not knowledge_base_exists():
        with st.spinner("Generating knowledge base documents..."):
            generate_knowledge_base()
        with st.spinner("Embedding and indexing documents (runs locally, no API call)..."):
            build_index()
        st.success("Knowledge base ready.")
        st.rerun()


def run_async(coro):
    """Run an async coroutine from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Setup")

    if knowledge_base_exists():
        st.success("✅ Knowledge base ready")
    else:
        st.warning("Knowledge base not built yet.")
        if st.button("⚙️ Build knowledge base"):
            generate_knowledge_base()
            build_index()
            st.success("Done!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Pipeline")
    st.markdown("""
1. **Researcher** — recalls past findings, searches knowledge base
2. **Summarizer** — condenses findings
3. **Writer** — drafts a structured report
4. **You** — approve or give feedback
5. **Finalize** — saves approved report
    """)
    st.markdown("---")
    max_revisions = st.slider("Max revisions", 1, 5, 2)

    if st.button("🗑️ Reset session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ── first-run guard ───────────────────────────────────────────────────────────

build_knowledge_base_if_needed()


# ── session state init ────────────────────────────────────────────────────────

if "phase" not in st.session_state:
    st.session_state.phase = "input"          # input | running | review | done
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
if "graph" not in st.session_state:
    st.session_state.graph = build_graph(st.session_state.checkpointer)
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "report" not in st.session_state:
    st.session_state.report = ""
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "revision_count" not in st.session_state:
    st.session_state.revision_count = 0
if "log" not in st.session_state:
    st.session_state.log = []


# ── phase: input ──────────────────────────────────────────────────────────────

if st.session_state.phase == "input":
    st.markdown("### Enter a research topic")
    topic = st.text_input(
        "Topic",
        placeholder="e.g. The state of quantum computing in 2026",
        key="topic_input",
    )
    if st.button("🚀 Start Research", type="primary", disabled=not topic.strip()):
        st.session_state.topic = topic.strip()
        st.session_state.phase = "running"
        st.rerun()


# ── phase: running ────────────────────────────────────────────────────────────

elif st.session_state.phase == "running":
    st.markdown(f"### 🔍 Researching: *{st.session_state.topic}*")

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    async def invoke_graph():
        return await st.session_state.graph.ainvoke(
            {
                "topic": st.session_state.topic,
                "findings": [],
                "summary": None,
                "report": None,
                "human_feedback": None,
                "is_approved": False,
                "revision_count": 0,
                "max_revisions": max_revisions,
                "final_report": None,
            },
            config=config,
        )

    with st.spinner("Agents are working — researcher → summarizer → writer..."):
        result = run_async(invoke_graph())

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.report = payload["report"]
        st.session_state.graph_state = result
        st.session_state.phase = "review"
        st.rerun()
    else:
        # Completed without needing review (shouldn't happen in normal flow)
        st.session_state.final_report = result.get("final_report", "")
        st.session_state.phase = "done"
        st.rerun()


# ── phase: review ─────────────────────────────────────────────────────────────

elif st.session_state.phase == "review":
    rev = st.session_state.revision_count + 1
    st.markdown(f"### 📝 Draft Report (revision {rev}) — *{st.session_state.topic}*")
    st.markdown("---")
    st.markdown(st.session_state.report)
    st.markdown("---")

    st.markdown("### 👀 Your Review")
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("✅ Approve", type="primary"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            async def resume_approve():
                return await st.session_state.graph.ainvoke(
                    Command(resume="approve"), config=config
                )

            with st.spinner("Finalizing report..."):
                result = run_async(resume_approve())

            st.session_state.final_report = result.get("final_report", st.session_state.report)
            st.session_state.phase = "done"
            st.rerun()

    with col2:
        feedback = st.text_input("Or give feedback for a revision:", placeholder="e.g. Add more detail on economic impacts")
        if st.button("🔄 Request revision", disabled=not feedback.strip()):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            async def resume_feedback():
                return await st.session_state.graph.ainvoke(
                    Command(resume=feedback.strip()), config=config
                )

            with st.spinner("Writer is revising..."):
                result = run_async(resume_feedback())

            st.session_state.revision_count += 1

            if "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                st.session_state.report = payload["report"]
                st.rerun()
            else:
                st.session_state.final_report = result.get("final_report", st.session_state.report)
                st.session_state.phase = "done"
                st.rerun()


# ── phase: done ───────────────────────────────────────────────────────────────

elif st.session_state.phase == "done":
    st.success("✅ Report approved and finalized!")
    st.markdown(f"### 📄 Final Report — *{st.session_state.topic}*")
    st.markdown("---")
    st.markdown(st.session_state.final_report or st.session_state.report)
    st.markdown("---")
    st.download_button(
        "⬇️ Download report as .txt",
        data=st.session_state.final_report or st.session_state.report,
        file_name=f"report_{st.session_state.topic[:40].replace(' ', '_')}.txt",
        mime="text/plain",
    )
    if st.button("🔁 Research another topic"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
