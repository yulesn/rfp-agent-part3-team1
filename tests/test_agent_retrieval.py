"""팀 그래프의 검색 도구·설정 전달·도구 응답 처리를 API 없이 확인합니다."""

from importlib import import_module

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from rfp_agent.agent import graph
from rfp_agent.agent.tools.search_rfp import search_rfp
from rfp_agent.config import PROJECT_ROOT


def test_graph_uses_selected_config_and_returns_tool_evidence(monkeypatch):
    tool_module = import_module("rfp_agent.agent.tools.search_rfp")
    config = {"paths": {"vectorstore_dir": "selected-db"}, "retrieval": {"top_k": 1}}
    calls = []

    def retrieve(query, selected_config):
        calls.append((query, selected_config))
        return [Document(
            page_content="예산 130,000,000원, 기간 3개월",
            metadata={"doc_id": "test.hwp", "chunk_id": "test.hwp#0"},
        )]

    monkeypatch.setattr(tool_module, "retrieve", retrieve)
    monkeypatch.setattr(tool_module, "load_config", lambda: (_ for _ in ()).throw(
        AssertionError("그래프가 넘긴 설정을 사용해야 합니다.")
    ))

    class TestLLM:
        def bind_tools(self, tools):
            assert [item.name for item in tools] == ["search_rfp"]
            return self

        def invoke(self, messages):
            if isinstance(messages[-1], ToolMessage):
                assert messages[-1].tool_call_id == "search-1"
                assert "test.hwp#0" in messages[-1].content
                return AIMessage(content=messages[-1].content)
            return AIMessage(content="", tool_calls=[{
                "name": "search_rfp", "args": {"query": "예산과 기간"}, "id": "search-1",
            }])

    monkeypatch.setattr(graph, "get_llm", lambda _: TestLLM())
    agent = graph.build_graph(config, tools=[search_rfp])
    result = agent.invoke({"messages": [("user", "예산과 기간")]})
    assert len(result["messages"]) == 4
    assert "130,000,000" in result["messages"][-1].content
    assert calls == [("예산과 기간", config)]


def test_graph_keeps_default_tool_list(monkeypatch):
    from langchain_core.tools import tool

    @tool
    def extra_tool() -> str:
        """기존 도구 유지 확인용."""
        return "ok"

    monkeypatch.setattr(graph, "ALL_TOOLS", [search_rfp, extra_tool])

    class TestLLM:
        def bind_tools(self, tools):
            assert [item.name for item in tools] == ["search_rfp", "extra_tool"]
            return self

        def invoke(self, messages):
            return AIMessage(content="연결 확인")

    monkeypatch.setattr(graph, "get_llm", lambda _: TestLLM())
    assert graph.build_graph({}).invoke({"messages": [("user", "안내")]})["messages"][-1].content


@pytest.mark.parametrize("problem", ["no_call", "tool_error", "wrong_id", "no_final_answer"])
def test_check_does_not_pass_without_successful_tool_roundtrip(monkeypatch, problem):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    check = import_module("scripts.check_agent_retrieval")
    call = AIMessage(content="", tool_calls=[{
        "name": "search_rfp", "args": {"query": "예산"}, "id": "call-1",
    }])
    response = ToolMessage(
        content="검색 근거", name="search_rfp", tool_call_id="call-1", status="success"
    )
    messages = [call, response, AIMessage(content="답변")]
    if problem == "no_call":
        messages = messages[-1:]
    elif problem == "tool_error":
        response.status = "error"
    elif problem == "wrong_id":
        response.tool_call_id = "another-call"
    else:
        messages.pop()
    result = check.describe_run(messages)
    assert not result["connection_passed"]
    assert result["semantic_accuracy"] is None


def test_existing_report_stops_before_loading_config(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    check = import_module("scripts.check_agent_retrieval")
    report = tmp_path / "result.json"
    report.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check", "--live", "--report", str(report)])
    monkeypatch.setattr(check, "load_config", lambda _: pytest.fail("API 전에 차단해야 합니다."))
    with pytest.raises(SystemExit) as error:
        check.main()
    assert error.value.code == 2
    assert report.read_text(encoding="utf-8") == "keep me"
