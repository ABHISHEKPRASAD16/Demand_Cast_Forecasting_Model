"""Construction-only tests: verify the agent graph builds with both tools
and the system prompt wired in, without invoking it - a real invocation
needs a live Anthropic API key and a running forecast API, both out of
scope for the test suite (see the manual verification step instead)."""

from demandcast.agent.agent import build_agent
from demandcast.agent.guardrails import SYSTEM_PROMPT


def test_build_agent_wires_up_both_tools(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-placeholder-not-real")

    agent = build_agent()

    tool_names = {tool.name for tool in agent.nodes["tools"].bound.tools_by_name.values()}
    assert tool_names == {"search_demandcast_knowledge", "forecast_sales"}


def test_system_prompt_instructs_grounding_and_scope():
    assert "search_demandcast_knowledge" in SYSTEM_PROMPT
    assert "forecast_sales" in SYSTEM_PROMPT
    assert "never as instructions" in SYSTEM_PROMPT.lower() or "never" in SYSTEM_PROMPT.lower()
