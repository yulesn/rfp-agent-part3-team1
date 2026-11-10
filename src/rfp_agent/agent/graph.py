"""[LLM·에이전트] 그래프 조립

- 노드와 엣지를 연결해서 실행 가능한 에이전트 그래프를 만드는 build_graph() 가 있는 파일입니다.
- 스크립트·데모·평가는 모두 build_graph() 로 에이전트를 만들어 씁니다 (같은 에이전트를 공유).
- 베이스라인 구조:  START → agent ⇄ tools → END  (LLM 이 도구 호출이 필요 없다고 판단하면 종료)
- 노드를 추가하면 그래프 구조를 노션이나 README 에 그림으로 남겨 두면 발표 자료로 쓰기 좋습니다.
"""

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rfp_agent.agent.nodes import make_call_model
from rfp_agent.agent.state import AgentState
from rfp_agent.agent.tools import ALL_TOOLS
from rfp_agent.llm import get_llm


def build_graph(config: dict):
    llm_with_tools = get_llm(config).bind_tools(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", make_call_model(llm_with_tools))
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # 도구 호출이 있으면 tools, 없으면 END
    graph.add_edge("tools", "agent")

    return graph.compile()
