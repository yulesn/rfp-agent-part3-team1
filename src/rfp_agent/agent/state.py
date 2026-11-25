"""[LLM·에이전트] 그래프 State 정의

- LangGraph 그래프의 노드들이 주고받는 상태(State)를 정의합니다.
- messages: 대화 전체(사용자 질문, 도구 호출·결과, 답변). 체크포인터(MemorySaver)를 붙이면 thread_id 별로 저장됩니다.
- current_doc_id: 지금 대화에서 다루는 공고 1건의 doc_id. 후속 질문의 기본 검색 범위로 씁니다 (2차 기능).
  도구가 공고 1건을 특정하면 track_doc 노드(nodes.py)가 채우고, 여러 공고를 다루면 비웁니다.
"""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_doc_id: NotRequired[str | None]
