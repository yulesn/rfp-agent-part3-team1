"""후속 질문 대화: thread_id 별 대화 유지, 대화 중인 공고(current_doc_id)로 검색 범위 고정,
이전 턴의 도구 결과 원문·오래된 턴 제외를 외부 서비스 없이 확인합니다."""

from importlib import import_module
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from rfp_agent.agent import graph
from rfp_agent.agent.nodes import TRACE_MARKER, history_view, track_doc
from rfp_agent.agent.tools.search_rfp import search_rfp
from rfp_agent.api import main as api

search_module = import_module("rfp_agent.agent.tools.search_rfp")
CONFIG = {"retrieval": {"top_k": 2}}


def _doc(doc_id, text="본문"):
    return Document(page_content=text, metadata={"doc_id": doc_id, "chunk_id": f"{doc_id}#0"})


class ScriptedLLM:
    """질문마다 정해 둔 도구 호출을 한 번 하고, 도구 결과가 오면 답합니다. 받은 입력을 기록합니다."""

    def __init__(self, plans):
        self.plans = list(plans)  # 질문마다 search_rfp 인자 (None 이면 도구 없이 바로 답)
        self.inputs = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"답변: {messages[-1].content[:40]}")
        args = self.plans.pop(0)
        if args is None:
            return AIMessage(content="도구 없이 답변")
        return AIMessage(content="", tool_calls=[{
            "name": "search_rfp", "args": args, "id": f"call-{len(self.inputs)}"}])


def _setup(monkeypatch, plans, results):
    calls = []

    def retrieve(query, config, filters=None):
        calls.append(filters)
        return results.pop(0)

    monkeypatch.setattr(search_module, "retrieve", retrieve)
    llm = ScriptedLLM(plans)
    monkeypatch.setattr(graph, "get_llm", lambda _: llm)
    agent = graph.build_graph(CONFIG, tools=[search_rfp], checkpointer=graph.new_memory())
    return agent, llm, calls


def test_follow_up_searches_inside_current_doc(monkeypatch):
    agent, llm, calls = _setup(
        monkeypatch,
        plans=[{"query": "서민금융 예산"}, {"query": "사업기간"}, {"query": "ERP", "search_all": True}],
        results=[[_doc("a.hwp"), _doc("a.hwp")], [_doc("a.hwp")], [_doc("b.hwp"), _doc("c.hwp")]],
    )
    thread = {"configurable": {"thread_id": "t1"}}

    first = agent.invoke({"messages": [("user", "서민금융 사업 예산은?")]}, thread)
    assert first["current_doc_id"] == "a.hwp"  # 검색 결과가 공고 1건 → 대화 중인 공고로
    assert calls == [None]

    second = agent.invoke({"messages": [("user", "기간은?")]}, thread)
    assert calls[-1] == {"doc_id": "a.hwp"}  # 후속 질문은 그 공고 안에서만 검색
    assert "a.hwp 안에서 검색" in second["messages"][-2].content
    system = llm.inputs[-1][0]
    assert isinstance(system, SystemMessage) and "a.hwp" in system.content

    third = agent.invoke({"messages": [("user", "다른 ERP 공고도 있어?")]}, thread)
    assert calls[-1] is None  # search_all=true 면 전체 검색
    assert third["current_doc_id"] is None  # 여러 공고 결과 → 특정 공고 대화 끝


def test_streamlit_displays_evidence_for_each_turn(monkeypatch):
    first_chunk = _doc("a.pdf", "예산 근거")
    follow_up_chunk = Document(
        page_content="참가 자격 근거",
        metadata={"doc_id": "a.pdf", "chunk_id": "a.pdf#2", "page_num": 7},
    )
    agent, _, calls = _setup(
        monkeypatch,
        plans=[{"query": "예산"}, {"query": "참가 자격"}, None],
        results=[[first_chunk, first_chunk], [follow_up_chunk]],
    )
    monkeypatch.setattr(api, "_graph", agent)
    client = TestClient(api.app, base_url="http://localhost:8000")
    monkeypatch.setattr(httpx, "get", lambda url, *, timeout: client.get(url))
    monkeypatch.setattr(httpx, "post", lambda url, *, json, timeout: client.post(url, json=json))

    path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(path), default_timeout=15).run()
    app.chat_input[0].set_value("공고의 예산은?").run()
    assert not app.exception
    assert not app.error
    assert app.expander[0].label == "검색된 원문 근거 보기 (1개)"
    assert app.expander[0].get("text")[0].value == "예산 근거"
    assert "쪽" not in app.expander[0].get("caption")[0].value

    app.chat_input[0].set_value("그럼 참가 자격은?").run()
    assert not app.exception
    assert not app.error
    assert calls == [None, {"doc_id": "a.pdf"}]
    assert len(app.expander) == 2
    assert [entry.value for entry in app.expander[1].get("text")] == ["참가 자격 근거"]
    assert "PDF 7쪽" in app.expander[1].get("caption")[0].value

    app.chat_input[0].set_value("고마워").run()
    assert not app.exception
    assert not app.error
    assert len(app.expander) == 2  # 검색하지 않은 답변에는 이전 근거를 붙이지 않습니다.


def test_threads_are_separate(monkeypatch):
    agent, _, calls = _setup(
        monkeypatch,
        plans=[{"query": "예산"}, {"query": "기간"}],
        results=[[_doc("a.hwp")], [_doc("b.hwp")]],
    )
    agent.invoke({"messages": [("user", "A 예산")]}, {"configurable": {"thread_id": "u1"}})
    other = agent.invoke({"messages": [("user", "기간")]}, {"configurable": {"thread_id": "u2"}})
    assert calls == [None, None]  # 다른 대화의 공고가 섞이지 않음
    assert other["current_doc_id"] == "b.hwp"
    assert [m.content for m in other["messages"] if isinstance(m, HumanMessage)] == ["기간"]


def test_history_keeps_previous_answers_without_tool_output(monkeypatch):
    agent, llm, _ = _setup(
        monkeypatch,
        plans=[{"query": "예산"}, None],
        results=[[_doc("a.hwp", "예산 원문 청크 230,000천원")]],
    )
    thread = {"configurable": {"thread_id": "t"}}
    agent.invoke({"messages": [("user", "예산은?")]}, thread)
    agent.invoke({"messages": [("user", "그럼 요약하면?")]}, thread)

    seen = llm.inputs[-1][1:]  # 시스템 프롬프트 제외
    assert [type(m).__name__ for m in seen] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert seen[1].content.startswith("답변:")  # 이전 최종 답변은 남김
    assert not any(isinstance(m, ToolMessage) for m in seen)  # 이전 도구 결과 원문은 뺌


def test_history_view_trims_old_turns_and_trace():
    messages = []
    for i in range(6):
        messages += [
            HumanMessage(f"q{i}"),
            AIMessage(content="", tool_calls=[{"name": "search_rfp", "args": {}, "id": f"c{i}"}]),
            ToolMessage(content="긴 청크", tool_call_id=f"c{i}"),
            AIMessage(content=f"a{i}{TRACE_MARKER} 기록...")]
    messages += [HumanMessage("now"),
                 AIMessage(content="", tool_calls=[{"name": "search_rfp", "args": {}, "id": "x"}]),
                 ToolMessage(content="이번 턴 결과", tool_call_id="x")]

    view = history_view(messages, max_turns=2)
    assert [m.content for m in view[:4]] == ["q4", "a4", "q5", "a5"]  # 최근 2턴, 기록(trace) 제거
    assert view[4].content == "now" and view[-1].content == "이번 턴 결과"  # 현재 턴은 그대로
    assert history_view(messages, max_turns=0)[0].content == "now"


def test_track_doc_rules():
    def state(*artifacts):
        return {"messages": [HumanMessage("q"), *[
            ToolMessage(content="r", tool_call_id=str(i), artifact=a) for i, a in enumerate(artifacts)]]}

    assert track_doc(state({"doc_ids": ["a", "a"]})) == {"current_doc_id": "a"}
    assert track_doc(state({"doc_ids": ["a"]}, {"doc_ids": ["b"]})) == {"current_doc_id": None}
    assert track_doc(state({"doc_ids": []})) == {}  # 결과 없음 → 그대로
    assert track_doc(state(None)) == {}  # doc_ids 를 안 주는 도구 → 그대로


def test_without_checkpointer_single_question_still_works(monkeypatch):
    monkeypatch.setattr(search_module, "retrieve", lambda q, c: [_doc("a.hwp")])
    monkeypatch.setattr(graph, "get_llm", lambda _: ScriptedLLM([{"query": "예산"}]))
    result = graph.build_graph(CONFIG, tools=[search_rfp]).invoke({"messages": [("user", "예산")]})
    assert result["messages"][-1].content.startswith("답변:")


def test_follow_up_after_summary_searches_that_doc(monkeypatch):
    summarize = import_module("rfp_agent.agent.tools.summarize_rfp")
    monkeypatch.setattr(summarize, "_summarize_doc",
                        lambda name, config, llm=None, current_doc_id=None: ("### 📄 서민금융 요약", "a.hwp"))
    calls = []
    monkeypatch.setattr(search_module, "retrieve",
                        lambda q, c, filters=None: calls.append(filters) or [_doc("a.hwp")])

    class LLM(ScriptedLLM):
        def invoke(self, messages):
            self.inputs.append(messages)
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="답변")
            name, args = self.plans.pop(0)
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "x"}])

    llm = LLM([("summarize_rfp", {"rfp_name": "서민금융"}), ("search_rfp", {"query": "하자보수"})])
    monkeypatch.setattr(graph, "get_llm", lambda _: llm)
    agent = graph.build_graph(CONFIG, tools=[search_rfp, summarize.summarize_rfp],
                              checkpointer=graph.new_memory())
    thread = {"configurable": {"thread_id": "s"}}
    assert agent.invoke({"messages": [("user", "서민금융 요약해줘")]}, thread)["current_doc_id"] == "a.hwp"
    agent.invoke({"messages": [("user", "하자보수 기간 자세히")]}, thread)
    assert calls == [{"doc_id": "a.hwp"}]  # 요약한 공고 안에서 후속 검색


def test_trace_is_attached_but_not_fed_back(monkeypatch):
    agent, llm, _ = _setup(monkeypatch, plans=[{"query": "예산"}, None], results=[[_doc("a.hwp")]])
    agent = graph.build_graph({**CONFIG, "agent": {"show_trace": True}}, tools=[search_rfp],
                              checkpointer=graph.new_memory())
    thread = {"configurable": {"thread_id": "t"}}
    first = agent.invoke({"messages": [("user", "예산은?")]}, thread)
    assert "실행 과정" in first["messages"][-1].content
    agent.invoke({"messages": [("user", "다시")]}, thread)
    previous_answer = llm.inputs[-1][2]
    assert previous_answer.content.startswith("답변:") and "실행 과정" not in previous_answer.content


def test_summarize_without_name_gets_current_doc(monkeypatch):
    summarize = import_module("rfp_agent.agent.tools.summarize_rfp")
    received = []
    monkeypatch.setattr(summarize, "_summarize_doc", lambda name, config, llm=None, current_doc_id=None:
                        received.append((name, current_doc_id)) or ("요약", current_doc_id))
    monkeypatch.setattr(search_module, "retrieve", lambda q, c, filters=None: [_doc("a.hwp")])

    class LLM(ScriptedLLM):
        def invoke(self, messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="답변")
            name, args = self.plans.pop(0)
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "x"}])

    monkeypatch.setattr(graph, "get_llm", lambda _: LLM([("search_rfp", {"query": "서민금융"}),
                                                         ("summarize_rfp", {})]))
    agent = graph.build_graph(CONFIG, tools=[search_rfp, summarize.summarize_rfp],
                              checkpointer=graph.new_memory())
    thread = {"configurable": {"thread_id": "s"}}
    agent.invoke({"messages": [("user", "서민금융 공고 찾아줘")]}, thread)
    agent.invoke({"messages": [("user", "그거 요약해줘")]}, thread)
    assert received == [("", "a.hwp")]
