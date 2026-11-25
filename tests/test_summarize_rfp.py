"""summarize_rfp: 공고 찾기, 항목별 검색 범위, '문서에 명시 없음' 처리를 API 없이 확인합니다."""

from importlib import import_module

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

summarize = import_module("rfp_agent.agent.tools.summarize_rfp")

CATALOG = [
    {"doc_id": "a.hwp", "발주 기관": "서민금융진흥원", "사업명": "서민금융 채팅 상담시스템 구축",
     "사업 금액": 230000000},
    {"doc_id": "b.hwp", "발주 기관": "국민연금공단", "사업명": "2024년 이러닝시스템 운영 용역"},
    {"doc_id": "c.hwp", "발주 기관": "국민연금공단", "사업명": "차세대 ERP 구축"},
]
CONFIG = {"paths": {"vectorstore_dir": "x"}, "vectorstore": {"collection_name": "rfp"},
          "retrieval": {"top_k": 5}}


def test_find_rfp_by_title_agency_and_ambiguity():
    assert summarize.find_rfp("서민금융 채팅 상담시스템 구축 사업 요약", CATALOG)[0]["doc_id"] == "a.hwp"
    assert [r["doc_id"] for r in summarize.find_rfp("국민연금공단 이러닝", CATALOG)] == ["b.hwp"]
    assert len(summarize.find_rfp("국민연금공단", CATALOG)) == 2
    assert summarize.find_rfp("전혀 없는 공고", CATALOG) == []


def test_list_markers_become_markdown_list():
    assert summarize._as_markdown("계약방법", "* 입찰방법: 제한경쟁\n• 협상에 의한 계약") == (
        "- **계약방법**\n  - 입찰방법: 제한경쟁\n  - 협상에 의한 계약")
    assert summarize._as_markdown("예산", "230,000천원") == "- **예산**: 230,000천원"


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def batch(self, inputs, config=None):
        self.prompts.extend(inputs)
        out = []
        for messages in inputs:
            body = messages[-1].content
            out.append(AIMessage(content="230,000천원" if "예산" in body.split("\n")[-1]
                                 else "문서에 명시 없음"))
        return out


def test_summary_uses_only_target_doc_and_marks_missing(monkeypatch):
    calls = []

    def retrieve(query, config, filters=None):
        calls.append((filters, config["retrieval"]["top_k"]))
        return [Document(page_content="사업예산 230,000천원", metadata={"chunk_id": "a.hwp#3"})]

    monkeypatch.setattr(summarize, "retrieve", retrieve)
    monkeypatch.setattr(summarize, "load_catalog", lambda config: CATALOG)
    monkeypatch.setattr(summarize, "_keyword_chunks", lambda field, doc_id, config: [])
    llm = FakeLLM()
    text = summarize._summarize("서민금융 채팅 상담시스템", CONFIG, llm=llm)

    for field in summarize.SUMMARY_FIELDS:
        assert f"- **{field}**" in text
    assert "- **발주기관**: 서민금융진흥원" in text  # 메타데이터에서 바로
    assert "- **예산**: 230,000천원" in text
    assert "- **하자보수**: 문서에 명시 없음" in text
    assert "a.hwp#3" not in text  # 사용자에게 청크 id 는 안 보임
    assert "공고 목록상 사업 금액: 230,000,000원" in llm.prompts[0][-1].content  # 예산엔 공고 목록 값도
    assert "공고 목록" not in llm.prompts[-1][-1].content  # 다른 항목엔 안 줌
    assert all(f == {"doc_id": "a.hwp"} for f, _ in calls)  # 다른 공고 청크는 안 섞임
    assert len(calls) == len(llm.prompts) == 9  # 메타데이터 2개 항목은 LLM 호출 안 함


def test_budget_keyword_chunk_and_metadata_answer_kept(monkeypatch):
    overview = Document(page_content="목차 ... □ 사업예산: 230,000천원 이내",
                        metadata={"chunk_id": "a.hwp#0", "chunk_index": 0})
    monkeypatch.setattr(summarize, "retrieve", lambda q, c, filters=None: [
        Document(page_content="입찰가격이 사업예산 이하인 자", metadata={"chunk_id": "a.hwp#60"})])
    monkeypatch.setattr(summarize, "load_catalog", lambda config: CATALOG)
    monkeypatch.setattr(summarize, "_keyword_chunks",
                        lambda field, doc_id, config: [overview] if field == "예산" else [])

    class AnswersFromList(FakeLLM):
        def batch(self, inputs, config=None):
            self.prompts.extend(inputs)
            return [AIMessage(content="문서에 명시 없음 (공고 목록상 사업 금액: 230,000,000원)"
                              if "공고 목록 정보" in m[-1].content else "문서에 명시 없음 (일반적으로 1년)")
                    for m in inputs]

    llm = AnswersFromList()
    text = summarize._summarize("서민금융 채팅 상담시스템", CONFIG, llm=llm)
    budget_prompt = llm.prompts[0][-1].content
    assert budget_prompt.index("a.hwp#0") < budget_prompt.index("a.hwp#60")  # 키워드 청크가 먼저
    # 공고 목록 값을 붙인 답은 그대로 두고, "명시 없음" 뒤 다른 설명은 버림
    assert "- **예산**: 문서에 명시 없음 (공고 목록상 사업 금액: 230,000,000원)" in text
    assert "- **하자보수**: 문서에 명시 없음\n" in text

def test_ambiguous_name_asks_back(monkeypatch):
    monkeypatch.setattr(summarize, "load_catalog", lambda config: CATALOG)
    text = summarize._summarize("국민연금공단", CONFIG, llm=FakeLLM())
    assert "여러 건" in text and "차세대 ERP 구축" in text


def test_graph_passes_config_to_summarize(monkeypatch):
    graph = import_module("rfp_agent.agent.graph")
    seen = []
    monkeypatch.setattr(summarize, "_summarize_doc",
                        lambda name, config, llm=None: (seen.append(config) or "ok", None))

    class TestLLM:
        def bind_tools(self, tools):
            assert "summarize_rfp" in [t.name for t in tools]
            return self

        def invoke(self, messages):
            if messages[-1].type == "tool":
                return AIMessage(content=messages[-1].content)
            return AIMessage(content="", tool_calls=[{
                "name": "summarize_rfp", "args": {"rfp_name": "서민금융"}, "id": "s-1"}])

    monkeypatch.setattr(graph, "get_llm", lambda _: TestLLM())
    agent = graph.build_graph(CONFIG, tools=[summarize.summarize_rfp])
    assert agent.invoke({"messages": [("user", "요약해줘")]})["messages"][-1].content == "ok"
    assert seen == [CONFIG]


def test_show_trace_records_every_llm_call_including_inside_summarize(monkeypatch):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    graph = import_module("rfp_agent.agent.graph")

    class AgentLLM(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    agent_llm = AgentLLM(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "summarize_rfp",
                                           "args": {"rfp_name": "서민금융"}, "id": "s-1"}]),
        AIMessage(content="최종 요약"),
    ]))
    inner_llm = GenericFakeChatModel(messages=iter([AIMessage(content="문서에 명시 없음")] * 20))
    monkeypatch.setattr(graph, "get_llm", lambda _: agent_llm)
    monkeypatch.setattr(summarize, "get_llm", lambda _: inner_llm)
    monkeypatch.setattr(summarize, "load_catalog", lambda config: CATALOG)
    monkeypatch.setattr(summarize, "retrieve", lambda q, c, filters=None: [
        Document(page_content="본문", metadata={"chunk_id": "a.hwp#1"})])
    monkeypatch.setattr(summarize, "_keyword_chunks", lambda field, doc_id, config: [])

    config = {**CONFIG, "agent": {"show_trace": True}}
    agent = graph.build_graph(config, tools=[summarize.summarize_rfp])
    answer = agent.invoke({"messages": [("user", "서민금융 요약")]})["messages"][-1].content
    assert answer.startswith("최종 요약")
    assert "LLM calls: 11" in answer  # agent 2번 + 요약 도구 안 9번
    assert "node=tools" in answer and "a.hwp#1" in answer  # 도구 안 프롬프트 전문
    assert "[output.tool_calls]" in answer

    plain = graph.build_graph(CONFIG, tools=[summarize.summarize_rfp])
    agent_llm.messages = iter([AIMessage(content="답")])
    assert plain.invoke({"messages": [("user", "안녕")]})["messages"][-1].content == "답"
