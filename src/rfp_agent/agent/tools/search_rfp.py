"""[RAG] RFP 내용 검색 도구 (1차 목표)

- 질문을 받아 retrieval/retriever.py 의 retrieve() 로 관련 청크를 찾고,
  LLM 이 읽기 좋은 텍스트(본문 + 출처)로 돌려줍니다.
- 도구의 docstring 은 LLM 이 "언제 이 도구를 쓸지" 판단하는 설명이라 구체적으로 써 주세요.
- 후속 질문(2차 기능): 그래프 안에서는 대화 중인 공고(state 의 current_doc_id)가 있으면 그 공고 안에서만
  검색합니다. LLM 이 search_all=true 를 주면 전체 공고에서 검색합니다.
  artifact의 doc_id 목록은 대화 중인 공고 추적에, 청크 본문과 출처는 화면 표시에 사용합니다.
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from rfp_agent.config import load_config
from rfp_agent.retrieval.retriever import retrieve


def _search(query: str, config: dict) -> str:
    return _format(retrieve(query, config))


def _format(docs) -> str:
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


@tool
def search_rfp(query: str) -> str:
    """RFP 문서 본문에서 질문과 관련된 내용을 검색합니다. 요구사항, 사업 내용, 제출 서류 등 문서 본문에 대한 질문에 사용하세요. 조건으로 공고 목록을 찾을 때는 filter_rfp_by_metadata 를 쓰세요."""
    return _search(query, load_config())


def make_search_rfp(config: dict):
    """에이전트가 선택한 DB·임베딩·top_k 설정을 검색 도구에도 전달합니다."""

    @tool("search_rfp", response_format="content_and_artifact", description=(
        "RFP 문서 본문에서 요구사항, 사업 내용, 평가 방법, 제출 서류 같은 내용의 근거와 출처를 검색합니다. "
        "발주기관·사업 금액·마감일 조건으로 공고 목록을 찾거나 세는 질문에는 쓰지 말고 filter_rfp_by_metadata 를 쓰세요. "
        "대화 중인 공고가 있으면 그 공고 안에서만 검색합니다. "
        "다른 공고나 여러 공고를 찾을 때만 search_all=true 로 전체 공고에서 검색하세요."))
    def configured_search(
        query: str,
        search_all: bool = False,
        state: Annotated[dict | None, InjectedState] = None,
    ) -> tuple[str, dict]:
        doc_id = None if search_all else (state or {}).get("current_doc_id")
        if doc_id:
            docs = retrieve(query, config, filters={"doc_id": doc_id})
            text = f"(대화 중인 공고 {doc_id} 안에서 검색한 결과)\n\n{_format(docs)}"
        else:
            docs = retrieve(query, config)
            text = _format(docs)
        return text, {
            "doc_ids": [doc.metadata.get("doc_id") for doc in docs],
            "evidence": [
                {
                    "source": doc.metadata.get("파일명") or doc.metadata.get("source")
                    or doc.metadata.get("doc_id") or "출처 미상",
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "page_num": doc.metadata.get("page_num"),
                    "text": doc.page_content,
                }
                for doc in docs
            ],
        }

    return configured_search
