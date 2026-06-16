import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.tools import search_knowledge_base, record_finding, recall_findings


def get_model_client(model="gpt-4o-mini", temperature=0.3):
    return OpenAIChatCompletionClient(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_researcher_agent():
    return AssistantAgent(
        name="researcher",
        model_client=get_model_client(temperature=0.2),
        tools=[search_knowledge_base, record_finding, recall_findings],
        system_message=(
            "You are a research agent. Use recall_findings first to check for relevant prior research on "
            "this topic, then use search_knowledge_base to gather information from the research knowledge "
            "base. Use record_finding to save useful new findings to persistent memory. Reply with a "
            "structured list of findings, each as a short claim with its source."
        ),
    )


def build_summarizer_agent():
    return AssistantAgent(
        name="summarizer",
        model_client=get_model_client(temperature=0.2),
        system_message=(
            "You are a summarization agent. Condense raw research findings into a concise, well-organized "
            "summary that highlights the most important points and removes redundancy."
        ),
    )


def build_writer_agent():
    return AssistantAgent(
        name="writer",
        model_client=get_model_client(temperature=0.5),
        system_message=(
            "You are a report-writing agent. Turn a research summary into a structured report with a "
            "title, an introduction, clearly labeled sections, and a conclusion. Address any reviewer "
            "feedback provided to you."
        ),
    )


async def run_agent_task(agent, task):
    result = await agent.run(task=task)
    return result.messages[-1].to_text()
