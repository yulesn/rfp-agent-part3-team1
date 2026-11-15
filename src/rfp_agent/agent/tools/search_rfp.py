"""[RAG] RFP 내용 검색 도구 (1차 목표)

- 질문을 받아 retrieval/retriever.py 의 retrieve() 로 관련 청크를 찾고,
  LLM 이 읽기 좋은 텍스트(본문 + 출처)로 돌려줍니다.
- 도구의 docstring 은 LLM 이 "언제 이 도구를 쓸지" 판단하는 설명이라 구체적으로 써 주세요.
"""

from langchain_core.tools import tool

from rfp_agent.config import load_config
from rfp_agent.retrieval.retriever import retrieve


@tool
def search_rfp(query: str) -> str:
    """RFP 문서에서 질문과 관련된 내용을 검색합니다. 요구사항, 사업 내용, 제출 서류 등 문서 본문에 대한 질문에 사용하세요."""
    docs = retrieve(query, load_config())
    if not docs:
        return "검색된 근거가 없습니다. 질문을 구체화하거나 검색 조건을 확인하세요."
    results = []
    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        source = metadata.get("source") or metadata.get("doc_id", "출처 미상")
        citation = f"[{i}] 출처: {source} | 청크: {metadata.get('chunk_id', '미상')}"
        if metadata.get("page_num") not in (None, ""):
            citation += f" | 페이지: {metadata['page_num']}"
        results.append(f"{citation}\n{doc.page_content}")
    return "\n\n".join(results)
