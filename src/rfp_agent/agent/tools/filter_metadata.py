"""[LLM·에이전트] 메타데이터 조건 검색 도구 (1차 목표)

- "예산 5억 이상", "OO부 발주", "이번 달 마감" 처럼 조건으로 공고를 찾는 도구입니다.
- 벡터 검색이 아니라 메타데이터(발주기관, 사업금액, 마감일 등)로 필터링해서 공고 목록을 돌려줍니다.
- 메타데이터 키 이름은 ingest/metadata.py 에서 정한 것과 맞춰 주세요.
"""

from langchain_core.tools import tool


@tool
def filter_rfp_by_metadata(
    agency: str | None = None,
    min_budget: int | None = None,
    max_budget: int | None = None,
    deadline_before: str | None = None,
) -> str:
    """발주기관, 사업 금액 범위, 입찰 마감일 같은 조건으로 RFP 공고 목록을 찾습니다."""
    # TODO: 메타데이터 목록에서 조건에 맞는 공고를 골라 사업명·발주기관·금액·마감일을 표 형태 문자열로 반환
    raise NotImplementedError
