"""[에이전트·도구] RFP 핵심 요약 도구 (1차 목표)

- 특정 공고 1건을 컨설턴트용 요약 형식(사업 개요, 발주기관, 예산, 기간, 주요 요구사항, 제출 방식·마감)으로 정리합니다.
- 문서 전체를 넣을지, 항목별로 검색해서 채울지는 실험으로 정하고 「의사결정 로그」에 남겨 주세요.
"""

from langchain_core.tools import tool


@tool
def summarize_rfp(doc_id: str) -> str:
    """특정 RFP 공고 하나의 핵심 정보(사업 개요, 예산, 기간, 요구사항, 제출 방식)를 요약합니다."""
    # TODO: doc_id 에 해당하는 문서 내용을 가져와 LLM 으로 정해진 형식에 맞춰 요약
    raise NotImplementedError
