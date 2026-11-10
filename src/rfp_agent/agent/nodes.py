"""[LLM·에이전트] 그래프 노드 함수

- LangGraph 그래프의 각 노드(상태를 받아 업데이트를 돌려주는 함수)를 작성합니다.
- 베이스라인은 call_model 노드 하나 + ToolNode 로 충분합니다 (graph.py 참고).
- 노드가 더 필요해지면(예: 질문 분류, 답변 검증) 여기에 추가하세요.
"""

from langchain_core.messages import SystemMessage

from rfp_agent.agent.prompts import SYSTEM_PROMPT
from rfp_agent.agent.state import AgentState


def make_call_model(llm_with_tools):
    """도구가 연결된 LLM 을 받아서, 대화를 보고 답하거나 도구를 호출하는 노드를 만듭니다."""

    def call_model(state: AgentState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    return call_model
