"""[LLM·에이전트] RFP 핵심 요약 도구 (1차 목표)

- 공고 1건을 11개 고정 항목(11/17·11/19 회의 결정)으로 요약합니다.
- 흐름: 공고 찾기(메타데이터) → 항목별로 그 문서 청크만 검색(의미 검색, 예산은 키워드 청크도)
  → 항목별 LLM 추출(병렬) → 합치기
- 예산·마감은 공고 목록 값도 프롬프트에 주고, 문서에 없을 때만 "문서에 명시 없음 (공고 목록상 …)"으로 쓰게 합니다.
- 문서에 근거가 없는 항목은 반드시 "문서에 명시 없음"으로 씁니다 (평가셋 must_not 확인 대상).
- 그래프 안에서는 요약한 공고의 doc_id 를 artifact 로 돌려줘서, 이어지는 후속 질문이 그 공고 안에서 검색됩니다.
- 반대로 대화 중인 공고(state 의 current_doc_id)가 있으면 "그거 요약해줘" 처럼 rfp_name 이 비었거나
  "이 공고" 같은 지시어일 때 그 공고를 요약합니다. 사업명이 여러 공고에 걸리면 대화 중인 공고를 고릅니다.
- 항목별 검색어·프롬프트를 바꾸면 노션 「실험 로그」에 결과를 남겨 주세요.
"""

import unicodedata
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from rfp_agent.agent.prompts import SUMMARY_FIELD_PROMPT
from rfp_agent.config import load_config
from rfp_agent.llm import get_llm
from rfp_agent.retrieval.retriever import retrieve
from rfp_agent.retrieval.vectorstore import load_vectorstore

NOT_FOUND = "문서에 명시 없음"

# 항목 이름 → (검색어, 추출 지침). 검색어는 그 문서 안에서 해당 내용이 있는 청크를 찾는 데 씁니다.
SUMMARY_FIELDS: dict[str, tuple[str, str]] = {
    "발주기관": ("발주기관 수요기관 기관명", "사업을 발주한 기관 이름"),
    "사업명": ("사업명 용역명", "공고된 사업(용역)의 정식 이름"),
    "예산": (
        "사업예산 추정가격 사업금액 부가세 포함 원",
        "사업 예산·추정가격 금액. 부가세 포함 여부와 별도 금액이 있으면 함께",
    ),
    "사업기간": ("사업기간 계약기간 계약체결일로부터 수행기간", "사업(계약) 수행 기간"),
    "계약방법": (
        "계약방법 입찰방식 협상에 의한 계약 제한경쟁 일반경쟁",
        "입찰·계약 방식 (예: 일반경쟁, 협상에 의한 계약)",
    ),
    "주요 요구사항": (
        "주요 과업 내용 사업 범위 요구사항 구축 기능",
        "핵심 과업·요구사항 3~5개를 짧은 항목으로",
    ),
    "참가 자격": (
        "입찰참가자격 참가 자격 중소기업 업종 등록 직접생산확인",
        "입찰 참가 자격 요건",
    ),
    "평가": (
        "제안서 평가 방법 기술평가 가격평가 배점 비율 협상적격자",
        "평가 방식과 기술·가격 배점, 협상적격 기준",
    ),
    "제출 마감": (
        "제안서 제출 마감 일시 접수 기한 제출 방법",
        "제안서·입찰서 제출 마감 일시. 다른 공고문을 참조하라고 되어 있으면 그 사실을",
    ),
    "하자보수": ("하자보수 무상 하자 담보 기간", "하자보수 기간과 조건"),
    "공동수급·하도급": (
        "공동수급 공동이행 분담이행 하도급 재하도급",
        "공동수급 허용 여부·방식과 하도급 허용 여부·제한",
    ),
}
# 메타데이터(제공받은 CSV)에서 바로 가져오는 항목. 키 이름은 팀 청크 메타데이터 기준
METADATA_FIELDS = {"발주기관": ("발주 기관", "발주기관"), "사업명": ("사업명",)}
# 공고 목록(메타데이터) 값을 프롬프트에 함께 주는 항목: 항목 → (메타데이터 키, 표시 이름)
# 문서에 값이 없을 때만 "문서에 명시 없음 (공고 목록상 입찰 마감: …)" 형식으로 쓰게 합니다 (SUMMARY_FIELD_PROMPT).
METADATA_IN_PROMPT = {"예산": ("사업 금액", "사업 금액"), "제출 마감": ("입찰 참여 마감일", "입찰 마감")}
# 의미 검색만으로는 잘 안 잡히는 항목은 키워드가 들어간 청크도 함께 넣습니다.
# 예) 예산은 문서 앞 "사업 개요" 청크(목차와 섞여 있음)에 있는데, "사업예산"이란 말이
#     평가·입찰 조건 청크에 더 자주 나와서 의미 검색 top-k 를 그쪽이 차지합니다.
KEYWORD_FIELDS = {"예산": ("사업예산", "사업 예산", "사업금액", "사업 금액", "추정가격", "사업비")}
KEYWORD_CHUNKS = 2
# rfp_name 이 이 말들(띄어쓰기 무시)이거나 비어 있으면 대화 중인 공고를 요약합니다.
CURRENT_DOC_WORDS = {"이공고", "그공고", "해당공고", "현재공고", "위공고", "이사업", "그사업", "해당사업",
                     "이거", "그거", "이것", "그것", "이rfp", "그rfp", "해당rfp"}
CHUNKS_PER_FIELD = 3

_catalog_cache: dict[tuple, list[dict]] = {}


def _compact(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", text).casefold() if c.isalnum())


def _first(record: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_catalog(config: dict) -> list[dict]:
    """벡터DB 메타데이터에서 공고 목록(doc_id별 메타데이터 1건)을 만듭니다. 본문·벡터는 읽지 않습니다."""
    key = (config["paths"]["vectorstore_dir"], config["vectorstore"]["collection_name"])
    if key not in _catalog_cache:
        records = load_vectorstore(config).get(include=["metadatas"])["metadatas"] or []
        by_doc = {}
        for record in records:
            if record and isinstance(record.get("doc_id"), str):
                by_doc.setdefault(record["doc_id"], record)
        _catalog_cache[key] = list(by_doc.values())
    return _catalog_cache[key]


def find_rfp(name: str, catalog: list[dict]) -> list[dict]:
    """사업명·기관명·파일명 일부로 공고를 찾습니다. 1건이면 확정, 여러 건이면 후보 목록입니다."""
    query = _compact(name)
    if not query:
        return []

    def title(record):
        return _compact(_first(record, METADATA_FIELDS["사업명"]) or "")

    def label(record):
        parts = [_first(record, k) or "" for k in METADATA_FIELDS.values()]
        return _compact(" ".join([*parts, record["doc_id"]]))

    # 1) doc_id·사업명이 그대로 들어 있으면 확정 (긴 사업명 우선)
    contained = [r for r in catalog if _compact(r["doc_id"]) == query
                 or (len(title(r)) >= 6 and title(r) in query)]
    if contained:
        return sorted(contained, key=lambda r: -len(title(r)))[:1]
    # 2) 띄어쓰기 단어가 가장 많이 들어 있는 공고 (단어 60% 이상 일치)
    words = [w for w in (_compact(w) for w in name.split()) if len(w) >= 2] or [query]
    scores = [sum(w in label(r) for w in words) / len(words) for r in catalog]
    best = max(scores, default=0)
    return [r for r, s in zip(catalog, scores) if s == best] if best >= 0.6 else []


def _format_chunks(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('chunk_id', '?')}]\n{d.page_content}" for d in docs)


def _metadata_hint(field: str, record: dict) -> str:
    """예산·마감의 공고 목록 값을 "[공고 목록 정보]" 블록으로 만듭니다. 값이 없으면 빈 문자열."""
    key, label = METADATA_IN_PROMPT.get(field, ("", ""))
    value = record.get(key) if key else None
    if isinstance(value, int) and value > 0:
        value = f"{value:,}원"
    elif not (isinstance(value, str) and value.strip()):
        return ""
    return f"[공고 목록 정보]\n공고 목록상 {label}: {str(value).strip()}\n\n"


def _keyword_chunks(field: str, doc_id: str, config: dict) -> list[Document]:
    """그 문서에서 항목 키워드가 들어간 청크를 문서 앞쪽부터 KEYWORD_CHUNKS 개 가져옵니다."""
    keywords = KEYWORD_FIELDS.get(field)
    if not keywords:
        return []
    found = load_vectorstore(config).get(
        where={"doc_id": doc_id},
        where_document={"$or": [{"$contains": k} for k in keywords]},
        include=["documents", "metadatas"],
    )
    docs = [Document(page_content=text, metadata=meta or {})
            for text, meta in zip(found["documents"], found["metadatas"])]
    docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))  # 사업 개요는 보통 문서 앞쪽
    return docs[:KEYWORD_CHUNKS]


def _merge(*groups: list[Document]) -> list[Document]:
    merged: dict = {}
    for docs in groups:
        for doc in docs:
            merged.setdefault(doc.metadata.get("chunk_id") or id(doc), doc)
    return list(merged.values())


def _extract_fields(doc_id: str, record: dict, config: dict, llm) -> dict[str, str]:
    """항목별 요약 값을 돌려줍니다. 근거 청크는 개발용 로그(실행 과정)에서 확인합니다."""
    search_config = {**config, "retrieval": {**config["retrieval"], "top_k": CHUNKS_PER_FIELD}}
    results: dict[str, str] = {}
    pending: list[tuple[str, list]] = []

    for field, (query, guide) in SUMMARY_FIELDS.items():
        meta_value = _first(record, METADATA_FIELDS.get(field, ()))
        if meta_value:
            results[field] = meta_value
            continue
        docs = _merge(_keyword_chunks(field, doc_id, config),
                      retrieve(query, search_config, filters={"doc_id": doc_id}))
        hint = _metadata_hint(field, record)
        if not docs and not hint:
            results[field] = NOT_FOUND
            continue
        messages = [
            SystemMessage(SUMMARY_FIELD_PROMPT.format(field=field, guide=guide, not_found=NOT_FOUND)),
            HumanMessage(f"{hint}[문서 발췌]\n{_format_chunks(docs)}\n\n'{field}' 항목:"),
        ]
        pending.append((field, messages))

    answers = llm.batch([m for _, m in pending], config={"max_concurrency": 4})
    for (field, _), answer in zip(pending, answers):
        text = (answer.content or "").strip()
        if not text or (NOT_FOUND in text and "공고 목록상" not in text):
            text = NOT_FOUND  # "명시 없음" 뒤에 붙은 설명은 버림
        results[field] = text
    return {field: results[field] for field in SUMMARY_FIELDS}


def _as_markdown(field: str, value: str) -> str:
    """항목 하나를 마크다운 목록으로 만듭니다. 여러 줄이면 하위 목록으로 넣습니다."""
    lines = [line.strip().lstrip("*•-·").strip() for line in value.splitlines() if line.strip()]
    if len(lines) <= 1:
        return f"- **{field}**: {lines[0] if lines else NOT_FOUND}"
    return f"- **{field}**\n" + "\n".join(f"  - {line}" for line in lines)


def format_summary(record: dict, fields: dict[str, str]) -> str:
    title = _first(record, METADATA_FIELDS["사업명"]) or record["doc_id"]
    source = _first(record, ("파일명",)) or record["doc_id"]
    items = "\n".join(_as_markdown(field, value) for field, value in fields.items())
    return f"### 📄 {title}\n\n출처 문서: {source}\n\n{items}"


def _pick_rfp(rfp_name: str, catalog: list[dict], current_doc_id: str | None) -> list[dict]:
    """요약할 공고 후보. 지시어·빈 이름이면 대화 중인 공고, 여러 건이면 그중 대화 중인 공고를 우선합니다."""
    current = [r for r in catalog if current_doc_id and r["doc_id"] == current_doc_id]
    if current and _compact(rfp_name) in CURRENT_DOC_WORDS | {""}:
        return current
    candidates = find_rfp(rfp_name, catalog)
    if len(candidates) > 1 and current and current[0] in candidates:
        return current
    return candidates


def _summarize_doc(rfp_name: str, config: dict, llm=None,
                   current_doc_id: str | None = None) -> tuple[str, str | None]:
    """(요약 텍스트, 요약한 공고 doc_id) 를 돌려줍니다. 공고를 1건으로 못 정하면 doc_id 는 None."""
    candidates = _pick_rfp(rfp_name, load_catalog(config), current_doc_id)
    if not candidates:
        return f"'{rfp_name}'에 해당하는 공고를 찾지 못했습니다. 사업명이나 발주기관을 더 정확히 알려 주세요.", None
    if len(candidates) > 1:
        names = "\n".join(f"- {_first(r, METADATA_FIELDS['사업명']) or r['doc_id']}"
                          for r in candidates[:5])
        return f"'{rfp_name}'에 해당하는 공고가 여러 건입니다. 어떤 공고인지 사용자에게 확인하세요.\n{names}", None
    record = candidates[0]
    fields = _extract_fields(record["doc_id"], record, config, llm or get_llm(config))
    return format_summary(record, fields), record["doc_id"]


def _summarize(rfp_name: str, config: dict, llm=None) -> str:
    return _summarize_doc(rfp_name, config, llm)[0]


_DESCRIPTION = (
    "RFP 공고 1건을 11개 항목(발주기관, 사업명, 예산, 사업기간, 계약방법, 주요 요구사항, "
    "참가 자격, 평가, 제출 마감, 하자보수, 공동수급·하도급)으로 요약합니다. "
    "사용자가 특정 공고의 요약·개요·핵심 정리를 원할 때 사용하세요. "
    "rfp_name 에는 사업명이나 발주기관 이름(일부도 가능)을 넣으세요. "
    "대화 중인 공고를 요약할 때는 rfp_name 을 비워 두세요."
)


@tool(description=_DESCRIPTION)
def summarize_rfp(rfp_name: str) -> str:
    return _summarize(rfp_name, load_config())


def make_summarize_rfp(config: dict):
    """에이전트가 쓰는 DB·LLM 설정을 요약 도구에도 전달합니다."""

    @tool("summarize_rfp", description=_DESCRIPTION, response_format="content_and_artifact")
    def configured_summarize(
        rfp_name: str = "",
        state: Annotated[dict | None, InjectedState] = None,
    ) -> tuple[str, dict]:
        current_doc_id = (state or {}).get("current_doc_id")
        text, doc_id = _summarize_doc(rfp_name, config, current_doc_id=current_doc_id)
        return text, {"doc_ids": [doc_id] if doc_id else []}  # track_doc 이 current_doc_id 로 저장

    return configured_summarize
