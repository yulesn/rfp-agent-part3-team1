"""후속 질문 대화: thread_id 별 대화 유지, 대화 중인 공고(current_doc_id)로 검색 범위 고정,
이전 턴의 도구 결과 원문·오래된 턴 제외를 API 없이 확인합니다."""

from importlib import import_module

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rfp_agent.agent import graph
from rfp_agent.agent.nodes import TRACE_MARKER, history_view, track_doc
from rfp_agent.agent.tools.search_rfp import search_rfp

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
