"""[LLM·에이전트] 그래프 State 정의

- LangGraph 그래프의 노드들이 주고받는 상태(State)를 정의합니다.
- 기본은 대화 메시지 목록만 있으면 됩니다. 노드가 늘어나면 필요한 필드를 추가하세요.
  예) 검색된 청크(context), 질문 유형(route), 대상 문서 ID 목록 등
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
