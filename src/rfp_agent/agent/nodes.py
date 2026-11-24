"""[LLM·에이전트] 그래프 노드 함수

- LangGraph 그래프의 각 노드(상태를 받아 업데이트를 돌려주는 함수)를 작성합니다.
- 베이스라인은 call_model 노드 하나 + ToolNode 로 충분합니다 (graph.py 참고).
- 후속 질문 대화(2차 기능):
  - call_model 은 LLM 에 대화 전체 대신 history_view() 로 줄인 대화를 넣습니다.
    이전 질문은 최근 N턴의 "질문 + 최종 답변"만 남기고 도구 결과 원문은 뺍니다 (Ollama 8k 컨텍스트).
  - track_doc 은 도구 결과에서 공고 1건이 특정되면 current_doc_id 에 저장합니다.
- 노드가 더 필요해지면(예: 질문 분류, 답변 검증) 여기에 추가하세요.
- 개발용: config 의 agent.show_trace 가 true 면 최종 답변 끝에 LLM 호출 기록 전체를 붙입니다 (trace.py).
"""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from rfp_agent.agent.prompts import CURRENT_DOC_NOTE, SYSTEM_PROMPT
from rfp_agent.agent.state import AgentState
from rfp_agent.agent.trace import format_calls

DEFAULT_HISTORY_TURNS = 4  # 현재 질문 외에 LLM 에 보여 줄 이전 대화 턴 수 (configs 의 agent.history_turns)
# 개발용 실행 기록(trace)이 답변 끝에 붙어 있으면 다음 턴 입력에서는 잘라 냅니다.
TRACE_MARKER = "\n\n---\n**🔍 실행 과정"


def _clean_answer(message: AIMessage) -> AIMessage:
    content = message.content if isinstance(message.content, str) else str(message.content)
    return AIMessage(content=content.split(TRACE_MARKER)[0])


def history_view(messages: list[AnyMessage], max_turns: int = DEFAULT_HISTORY_TURNS) -> list:
    """LLM 에 넣을 대화를 만듭니다.

    - 턴 = 사용자 질문 1개부터 다음 질문 전까지
    - 현재 턴(마지막 질문 이후)은 도구 호출·결과까지 그대로 둡니다 (ReAct 루프에 필요).
    - 이전 턴은 질문과 최종 답변만 남기고, 최근 max_turns 개만 남깁니다.
    """
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not starts:
        return list(messages)
    past, current = messages[: starts[-1]], messages[starts[-1]:]

    turns, turn = [], []
    for message in past:
        if isinstance(message, HumanMessage):
            if turn:
                turns.append(turn)
            turn = [message]
        elif isinstance(message, AIMessage) and not message.tool_calls and message.content:
            turn = [*turn[:1], _clean_answer(message)]  # 턴의 마지막 답변만
    if turn:
        turns.append(turn)

    kept = turns[-max_turns:] if max_turns > 0 else []
    return [m for t in kept for m in t] + list(current)


def make_call_model(llm_with_tools, *, history_turns: int = DEFAULT_HISTORY_TURNS, recorder=None):
    """도구가 연결된 LLM 을 받아서, 대화를 보고 답하거나 도구를 호출하는 노드를 만듭니다.

    recorder(LLMCallRecorder)가 있으면 최종 답변 끝에 이번 질문의 LLM 호출 기록 전체를 붙입니다.
    """

    def call_model(state: AgentState) -> dict:
        system = SYSTEM_PROMPT
        if state.get("current_doc_id"):
            system += CURRENT_DOC_NOTE.format(doc_id=state["current_doc_id"])
        messages = [SystemMessage(system), *history_view(state["messages"], history_turns)]
        response = llm_with_tools.invoke(messages)
        if recorder is not None and not response.tool_calls:
            content = response.content if isinstance(response.content, str) else str(response.content)
            response = response.model_copy(
                update={"content": content + format_calls(recorder.snapshot())})
        return {"messages": [response]}

    return call_model


def track_doc(state: AgentState) -> dict:
    """방금 실행된 도구 결과의 artifact {"doc_ids": [...]} 로 current_doc_id 를 갱신합니다.

    - 결과가 공고 1건에서만 나왔으면 그 공고로 정합니다.
    - 여러 공고에 걸쳐 있으면(전체 검색·조건 검색) 특정 공고 대화가 끝난 것으로 보고 비웁니다.
    - doc_ids 를 알려 주지 않는 도구는 current_doc_id 를 바꾸지 않습니다.
    """
    doc_ids: list[str] = []
    reported = False
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break  # 이번 도구 실행 결과만 봅니다
        artifact = message.artifact if isinstance(message.artifact, dict) else {}
        if "doc_ids" in artifact:
            reported = True
            doc_ids.extend(d for d in artifact["doc_ids"] if d)
    if not reported:
        return {}
    unique = set(doc_ids)
    if len(unique) == 1:
        return {"current_doc_id": unique.pop()}
    return {"current_doc_id": None} if unique else {}
