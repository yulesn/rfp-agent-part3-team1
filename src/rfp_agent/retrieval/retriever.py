"""[RAG] 리트리버 (팀 공통 약속)

- 에이전트 도구가 부르는 검색 함수입니다. 입력: 질문(str) / 출력: 관련 청크 목록(list[Document]).
- 이 함수의 입출력 형식은 LLM·에이전트 파트와의 약속이라, 바꿀 때는 팀에 먼저 공유해 주세요.
- 실험 아이디어: top_k 조정, 메타데이터 필터, MMR, BM25 + 벡터 하이브리드 검색, 리랭커 추가.
"""

import re
import unicodedata
from collections import OrderedDict
from threading import Lock

from chromadb.api.types import validate_where
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rfp_agent.retrieval.vectorstore import IndexIdentity, load_vectorstore_with_identity


_DETAIL_TERMS = (
    "예산", "금액", "사업기간", "과업기간", "계약기간", "시작일", "종료일", "일정",
    "업종", "자격", "등록기한", "마감", "평가", "배점", "비중", "비율", "요약",
    "계약방법", "입찰방식", "요구사항", "제출서류", "하도급", "하자보수",
)
_BROAD_TERMS = (
    "비슷", "유사", "다른", "타기관", "비교", "빼고", "제외", "말고", "목록",
    "나열", "들중", "모든공고", "관련공고", "다른사업",
)
_AGENCY_SUFFIX = r"의|은|는|이|가|을|를|에서|에게|과|와|도|만|으로|로"
_CATALOG_LIMIT = 8
NoticeCatalog = tuple[tuple[tuple[str, str, str], ...], frozenset[str]]
_catalogs: OrderedDict[IndexIdentity, NoticeCatalog] = OrderedDict()
_catalog_lock = Lock()


def _compact(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFC", text).casefold() if char.isalnum())


def _named_agency(agency: str, query: str) -> bool:
    # 기관명 끝에 붙은 조사는 허용하고, 한 기관명의 일부만 일치하는 경우는 제외한다.
    name = re.escape(unicodedata.normalize("NFC", agency).casefold())
    return bool(re.search(rf"(?<!\w){name}(?=$|\W|(?:{_AGENCY_SUFFIX})(?!\w))", query))


def _build_catalog(store: Chroma) -> NoticeCatalog:
    metadata = store.get(include=["metadatas"])["metadatas"] or []
    notices = set()
    agencies = set()
    for record in metadata:
        if not record:
            continue
        agency = record.get("발주 기관")
        if isinstance(agency, str) and agency.strip():
            agencies.add(agency)
        else:
            agency = ""
        title, doc_id = record.get("사업명"), record.get("doc_id")
        if isinstance(title, str) and isinstance(doc_id, str) and doc_id:
            notices.add((title, doc_id, agency))
    titles = tuple(
        (title, doc_id, agency)
        for raw_title, doc_id, agency in notices
        if len(title := _compact(raw_title)) >= 10
    )
    return titles, frozenset(agencies)


def _notice_catalog(store: Chroma, identity: IndexIdentity) -> NoticeCatalog:
    # 완성된 인덱스는 그대로 사용한다. 재구축한 컬렉션은 새 ID를 가지므로 새 목록을 읽는다.
    with _catalog_lock:
        if identity not in _catalogs:
            _catalogs[identity] = _build_catalog(store)
            if len(_catalogs) > _CATALOG_LIMIT:
                _catalogs.popitem(last=False)
        _catalogs.move_to_end(identity)
        return _catalogs[identity]


def _title_filter(query: str, store: Chroma, identity: IndexIdentity) -> dict[str, str] | None:
    compact_query = _compact(query)
    if not any(term in compact_query for term in _DETAIL_TERMS):
        return None

    notices, agencies = _notice_catalog(store, identity)
    matches = [notice for notice in notices if notice[0] in compact_query]
    if len(matches) != 1:
        return None

    title, doc_id, agency = matches[0]
    remainder = compact_query.replace(title, "")
    # 제목이 참고나 제외 대상으로 쓰인 경우 기존 전체 검색을 유지한다.
    if any(term in remainder for term in _BROAD_TERMS):
        return None
    if not any(term in remainder for term in _DETAIL_TERMS):
        return None
    normalized_query = unicodedata.normalize("NFC", query).casefold()
    if any(name != agency and _named_agency(name, normalized_query) for name in agencies):
        return None
    return {"doc_id": doc_id}


def retrieve(query: str, config: dict, filters: dict | None = None) -> list[Document]:
    """관련 청크를 반환합니다. filters는 Chroma where 문법, 조건 불일치는 []입니다.

    예: {"발주기관": "OO시청"}, {"사업금액": {"$gte": 500000000}}
    여러 조건은 {"$and": [{...}, {...}]} 형태로 전달하세요.
    filters를 생략한 단일 공고 상세 질문은 사업명으로 범위를 좁힙니다.
    retrieval.scope_by_title: false 설정으로 기존 전체 검색을 사용할 수 있습니다.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query는 비어 있지 않은 문자열이어야 합니다.")
    top_k = config["retrieval"]["top_k"]
    if type(top_k) is not int or top_k < 1:
        raise ValueError("retrieval.top_k는 1 이상의 정수여야 합니다.")
    explicit_filters = filters is not None
    if filters == {}:
        filters = None
    if filters is not None:
        validate_where(filters)
    vs, identity = load_vectorstore_with_identity(config)
    if not explicit_filters and config["retrieval"].get("scope_by_title", True):
        filters = _title_filter(query, vs, identity)
    return vs.similarity_search(query.strip(), k=top_k, filter=filters)
