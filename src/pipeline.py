import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from src.graph import build_graph

DB_PATH = "memory_store/checkpoints.sqlite"


async def run_research_pipeline(topic, max_revisions=2, human_input_fn=input, db_path=DB_PATH):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer)

        state = await graph.ainvoke(
            {
                "topic": topic,
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

        while "__interrupt__" in state:
            payload = state["__interrupt__"][0].value
            print("\n--- Human review checkpoint ---")
            print(payload["report"])
            print(payload["instructions"])
            response = human_input_fn("Your response: ")
            state = await graph.ainvoke(Command(resume=response), config=config)

    return state


if __name__ == "__main__":
    import sys

    topic = sys.argv[1] if len(sys.argv) > 1 else "The state of quantum computing in 2026"
    final_state = asyncio.run(run_research_pipeline(topic))
    print("\n--- Final Report ---")
    print(final_state["final_report"])
