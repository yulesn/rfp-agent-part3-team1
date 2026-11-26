"""filter_rfp_by_metadata: 발주기관·금액·마감일 조건 검색, 정렬·개수 제한, 1건이면 대화 중인 공고로 설정을 API 없이 확인합니다."""

from importlib import import_module

from langchain_core.messages import AIMessage, ToolMessage

from rfp_agent.agent import graph
from rfp_agent.agent.tools.filter_metadata import filter_rfp_by_metadata

module = import_module("rfp_agent.agent.tools.filter_metadata")

CATALOG = [
    {"doc_id": "a.hwp", "발주 기관": "국민연금공단", "사업명": "2024년 이러닝시스템 운영 용역",
     "사업 금액": 150_000_000, "입찰 참여 마감일": "2024-04-09 15:00:00"},
    {"doc_id": "b.hwp", "발주 기관": "국민연금공단", "사업명": "차세대 ERP 구축",
     "사업 금액": 1_200_000_000, "입찰 참여 마감일": "2024-05-30 10:00:00"},
    {"doc_id": "c.hwp", "발주 기관": "한국철도공사 (용역)", "사업명": "철도 AI 안전관리 시스템",
     "사업 금액": 500_000_000, "입찰 참여 마감일": "2024-04-30 18:00:00"},
    {"doc_id": "d.hwp", "발주 기관": "인천광역시", "사업명": "스마트시티 플랫폼", "사업 금액": 0},
]


def ids(**kwargs):
    records, total = module.filter_rfps(CATALOG, **kwargs)
    return [r["doc_id"] for r in records], total


def test_agency_and_keyword_partial_match():
    assert ids(agency="국민연금") == (["a.hwp", "b.hwp"], 2)
    assert ids(agency="한국 철도 공사") == (["c.hwp"], 1)  # 띄어쓰기·괄호 무시
    assert ids(keyword="erp") == (["b.hwp"], 1)


def test_budget_range_skips_unknown_budget():
    assert ids(min_budget=500_000_000) == (["c.hwp", "b.hwp"], 2)
    assert ids(max_budget=500_000_000) == (["a.hwp", "c.hwp"], 2)  # 금액 0(미상)은 제외
    assert ids(min_budget=0)[1] == 4  # LLM 이 0 을 넣으면 조건 없음


def test_deadline_range_includes_whole_end_day():
    assert ids(deadline_from="2024-04-01", deadline_to="2024-04-30") == (["a.hwp", "c.hwp"], 2)
    assert ids(deadline_to="2024-04-09") == (["a.hwp"], 1)  # 15:00 마감도 그날에 포함


def test_sort_and_limit():
    assert ids(sort_by="budget", descending=True, limit=2) == (["b.hwp", "c.hwp"], 4)
    assert ids()[0][-1] == "d.hwp"  # 마감일 없는 공고는 맨 뒤


def test_tool_text_and_bad_date(monkeypatch):
    monkeypatch.setattr(module, "load_catalog", lambda config: CATALOG)
    text, artifact = module._filter({}, agency="국민연금", sort_by="budget", descending=True, limit=1)
    assert "전체 2건 중 1건" in text
    assert "| 1 | 차세대 ERP 구축 | 국민연금공단 | 1,200,000,000원 | 2024-05-30 10:00:00 |" in text
    assert artifact == {"doc_ids": ["b.hwp"]}
    assert "공고가 없습니다" in module._filter({}, agency="없는기관")[0]
    text, artifact = module._filter({}, deadline_to="다음 주")
    assert "날짜 형식" in text and artifact == {}


class ScriptedLLM:
    def __init__(self, args):
        self.args = args

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=messages[-1].content)
        return AIMessage(content="", tool_calls=[
            {"name": "filter_rfp_by_metadata", "args": self.args, "id": "call-1"}])


def _run(monkeypatch, args):
    monkeypatch.setattr(module, "load_catalog", lambda config: CATALOG)
    monkeypatch.setattr(graph, "get_llm", lambda _: ScriptedLLM(args))
    agent = graph.build_graph({}, tools=[filter_rfp_by_metadata], checkpointer=graph.new_memory())
    return agent.invoke({"messages": [("user", "질문")]}, {"configurable": {"thread_id": "t"}})


def test_single_result_becomes_current_doc(monkeypatch):
    state = _run(monkeypatch, {"keyword": "ERP"})
    assert state["current_doc_id"] == "b.hwp"
    assert "차세대 ERP 구축" in state["messages"][-1].content
    assert "전체 1건 중 1건" in state["messages"][-1].content  # 건수가 다 나와도 표시


def test_multiple_results_clear_current_doc(monkeypatch):
    assert _run(monkeypatch, {"agency": "국민연금"})["current_doc_id"] is None
