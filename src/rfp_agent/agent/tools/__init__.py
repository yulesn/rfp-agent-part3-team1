"""[에이전트·도구] 에이전트가 쓰는 도구 목록

- 도구 1개 = 파일 1개 원칙입니다. 새 도구를 만들면 이 파일의 ALL_TOOLS 에 추가해 주세요.
- 1차 목표 도구: search_rfp(본문 검색), filter_rfp_by_metadata(조건 검색), summarize_rfp(공고 요약)
- 2차 목표로 추가할 도구는 팀 회의 후 결정
"""

from rfp_agent.agent.tools.filter_metadata import filter_rfp_by_metadata
from rfp_agent.agent.tools.search_rfp import search_rfp
from rfp_agent.agent.tools.summarize_rfp import summarize_rfp

ALL_TOOLS = [search_rfp, filter_rfp_by_metadata, summarize_rfp]
