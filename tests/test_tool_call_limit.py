"""도구 호출 상한(agent.max_tool_calls): 병렬 호출을 남은 횟수만큼 자르고, 다 쓰면 도구 없이 답하는지 확인합니다."""

from importlib import import_module

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rfp_agent.agent import graph
from rfp_agent.agent.prompts import TOOL_LIMIT_NOTE
from rfp_agent.agent.tools.search_rfp import search_rfp

search_module = import_module("rfp_agent.agent.tools.search_rfp")


class EndlessSearchLLM:
    """도구가 연결되면 매번 search_rfp 를 4개씩 병렬로 부르고, 도구 없이 불리면 답합니다."""

    def __init__(self):
        self.tool_rounds = 0
        self.plain_inputs = []

    def bind_tools(self, tools):
        outer = self

        class WithTools:
            def invoke(self, messages):
                outer.tool_rounds += 1
                calls = [{"name": "search_rfp", "args": {"query": f"q{outer.tool_rounds}-{i}"},
                          "id": f"call-{outer.tool_rounds}-{i}"} for i in range(4)]
                return AIMessage(content="", tool_calls=calls)

        return WithTools()

    def invoke(self, messages):
        self.plain_inputs.append(messages)
        return AIMessage(content="찾은 근거로 답변")


def _run(monkeypatch, agent_config, invoke_config=None):
    monkeypatch.setattr(search_module, "retrieve", lambda query, config, filters=None: [
        Document(page_content="본문", metadata={"doc_id": "a.hwp", "chunk_id": "a.hwp#0"})])
    llm = EndlessSearchLLM()
    monkeypatch.setattr(graph, "get_llm", lambda _: llm)
    agent = graph.build_graph({"retrieval": {"top_k": 1}, "agent": agent_config}, tools=[search_rfp])
    return agent.invoke({"messages": [("user", "질문")]}, invoke_config), llm


def test_stops_at_max_tool_calls(monkeypatch):
    result, llm = _run(monkeypatch, {"max_tool_calls": 6})
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 6  # 4개 + (남은 2개만) 2개
    assert llm.tool_rounds == 2
    assert result["messages"][-1].content == "찾은 근거로 답변"
    assert TOOL_LIMIT_NOTE in llm.plain_inputs[-1][0].content


def test_zero_means_no_limit(monkeypatch):
    # 상한이 없으면 recursion_limit 까지 계속 도구를 부릅니다.
    # (LangGraph 기본 recursion_limit 은 10007 이라 테스트에서는 작게 줍니다)
    import pytest
    from langgraph.errors import GraphRecursionError

    with pytest.raises(GraphRecursionError):
        _run(monkeypatch, {"max_tool_calls": 0}, {"recursion_limit": 12})
