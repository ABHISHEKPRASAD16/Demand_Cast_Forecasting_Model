"""Builds and runs the DemandCast AI Analyst agent: Claude reasoning over
two tools (retrieval against the RAG knowledge base, live forecast calls),
with the guardrails from guardrails.py applied to every response.
"""

import os

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from demandcast.agent.guardrails import SYSTEM_PROMPT, apply_output_guardrails
from demandcast.agent.tools import DEFAULT_API_BASE_URL, build_forecast_tool, build_retriever_tool

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def build_agent(api_base_url: str = DEFAULT_API_BASE_URL, model_name: str | None = None):
    model = ChatAnthropic(
        model=model_name or os.environ.get("DEMANDCAST_AGENT_MODEL", DEFAULT_MODEL)
    )
    tools = [build_retriever_tool(), build_forecast_tool(api_base_url)]
    return create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


def ask(agent, question: str) -> str:
    """Invoke the agent on one question and return its guardrail-checked final answer."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_message = result["messages"][-1]
    return apply_output_guardrails(final_message.content)
