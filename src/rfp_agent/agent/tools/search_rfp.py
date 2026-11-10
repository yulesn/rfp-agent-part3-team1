"""[RAG] RFP 내용 검색 도구 (1차 목표)

- 질문을 받아 retrieval/retriever.py 의 retrieve() 로 관련 청크를 찾고,
  LLM 이 읽기 좋은 텍스트(본문 + 출처)로 돌려줍니다.
- 도구의 docstring 은 LLM 이 "언제 이 도구를 쓸지" 판단하는 설명이라 구체적으로 써 주세요.
"""

from langchain_core.tools import tool


@tool
def search_rfp(query: str) -> str:
    """RFP 문서에서 질문과 관련된 내용을 검색합니다. 요구사항, 사업 내용, 제출 서류 등 문서 본문에 대한 질문에 사용하세요."""
    # TODO: config 로드 → retrieve(query, config) → 청크 내용과 출처(metadata)를 문자열로 합쳐서 반환
    raise NotImplementedError
