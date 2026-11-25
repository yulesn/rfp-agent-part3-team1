"""[LLM·에이전트] 그래프 조립

- 노드와 엣지를 연결해서 실행 가능한 에이전트 그래프를 만드는 build_graph() 가 있는 파일입니다.
- 스크립트·데모·평가는 모두 build_graph() 로 에이전트를 만들어 씁니다 (같은 에이전트를 공유).
- 구조:  START → agent ⇄ tools → track_doc → agent … → END  (LLM 이 도구 호출이 필요 없다고 판단하면 종료)
  track_doc 은 도구 결과로 대화 중인 공고(current_doc_id)를 갱신합니다 (후속 질문 대화).
- 대화를 이어가려면 checkpointer 를 넘기고 같은 thread_id 로 호출하세요. (new_memory() 참고)
    graph = build_graph(config, checkpointer=new_memory())
    graph.invoke({"messages": [("user", 질문)]}, {"configurable": {"thread_id": "대화ID"}})
  checkpointer 없이 부르면 지금처럼 질문 1개짜리 실행입니다 (평가 스크립트 등).
- 노드를 추가하면 그래프 구조를 노션이나 README 에 그림으로 남겨 두면 발표 자료로 쓰기 좋습니다.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rfp_agent.agent.nodes import DEFAULT_HISTORY_TURNS, make_call_model, track_doc
from rfp_agent.agent.state import AgentState
from rfp_agent.agent.tools import ALL_TOOLS
from rfp_agent.agent.tools.search_rfp import make_search_rfp, search_rfp
from rfp_agent.agent.tools.summarize_rfp import make_summarize_rfp, summarize_rfp
from rfp_agent.agent.trace import LLMCallRecorder
from rfp_agent.llm import get_llm


def new_memory():
    """대화 기록을 메모리에 저장하는 체크포인터 (프로세스가 끝나면 사라짐)."""
    return InMemorySaver()


def build_graph(config: dict, *, tools=None, checkpointer=None):
    """기본 도구 목록을 사용하며 연결 시험에서는 tools로 준비된 도구만 지정합니다.

    checkpointer 를 넘기면 thread_id 별로 대화(messages, current_doc_id)가 이어집니다.
    """
    selected = ALL_TOOLS if tools is None else tools
    # LLM과 도구가 서로 다른 설정 파일을 읽지 않도록 같은 config를 전달합니다.
    factories = [(search_rfp, make_search_rfp), (summarize_rfp, make_summarize_rfp)]
    configured = [
        next((make(config) for base, make in factories if item is base), item) for item in selected
    ]
    llm_with_tools = get_llm(config).bind_tools(configured)

    graph = StateGraph(AgentState)
    history_turns = config.get("agent", {}).get("history_turns", DEFAULT_HISTORY_TURNS)
    # 개발용: 모든 LLM 호출(도구 안의 호출 포함)을 기록해 최종 답변 끝에 붙입니다.
    recorder = LLMCallRecorder() if config.get("agent", {}).get("show_trace", False) else None
    graph.add_node("agent", make_call_model(
        llm_with_tools, history_turns=history_turns, recorder=recorder))
    graph.add_node("tools", ToolNode(configured))
    graph.add_node("track_doc", track_doc)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # 도구 호출이 있으면 tools, 없으면 END
    graph.add_edge("tools", "track_doc")
    graph.add_edge("track_doc", "agent")

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled.with_config(callbacks=[recorder]) if recorder else compiled
