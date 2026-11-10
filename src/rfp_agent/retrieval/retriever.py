"""[RAG] 리트리버 (팀 공통 약속)

- 에이전트 도구가 부르는 검색 함수입니다. 입력: 질문(str) / 출력: 관련 청크 목록(list[Document]).
- 이 함수의 입출력 형식은 LLM·에이전트 파트와의 약속이라, 바꿀 때는 팀에 먼저 공유해 주세요.
- 실험 아이디어: top_k 조정, 메타데이터 필터, MMR, BM25 + 벡터 하이브리드 검색, 리랭커 추가.
"""

from langchain_core.documents import Document

from rfp_agent.retrieval.vectorstore import load_vectorstore


def retrieve(query: str, config: dict, filters: dict | None = None) -> list[Document]:
    """질문과 관련된 청크를 top_k 개 돌려줍니다. filters 예: {"발주기관": "OO시청"}"""
    # TODO: 아래는 가장 단순한 버전. 실험하며 개선하세요.
    vs = load_vectorstore(config)
    return vs.similarity_search(query, k=config["retrieval"]["top_k"], filter=filters)
