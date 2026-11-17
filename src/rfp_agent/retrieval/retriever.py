"""[RAG] 리트리버 (팀 공통 약속)

- 에이전트 도구가 부르는 검색 함수입니다. 입력: 질문(str) / 출력: 관련 청크 목록(list[Document]).
- 이 함수의 입출력 형식은 LLM·에이전트 파트와의 약속이라, 바꿀 때는 팀에 먼저 공유해 주세요.
- 실험 아이디어: top_k 조정, 메타데이터 필터, MMR, BM25 + 벡터 하이브리드 검색, 리랭커 추가.
"""

from chromadb.api.types import validate_where
from langchain_core.documents import Document

from rfp_agent.retrieval.vectorstore import load_vectorstore


def retrieve(query: str, config: dict, filters: dict | None = None) -> list[Document]:
    """관련 청크를 반환합니다. filters는 Chroma where 문법, 조건 불일치는 []입니다.

    예: {"발주기관": "OO시청"}, {"사업금액": {"$gte": 500000000}}
    여러 조건은 {"$and": [{...}, {...}]} 형태로 전달하세요.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query는 비어 있지 않은 문자열이어야 합니다.")
    top_k = config["retrieval"]["top_k"]
    if type(top_k) is not int or top_k < 1:
        raise ValueError("retrieval.top_k는 1 이상의 정수여야 합니다.")
    if filters == {}:
        filters = None
    if filters is not None:
        validate_where(filters)
    vs = load_vectorstore(config)
    return vs.similarity_search(query.strip(), k=top_k, filter=filters)
