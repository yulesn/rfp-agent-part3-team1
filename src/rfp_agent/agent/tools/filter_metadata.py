"""[LLM·에이전트] 메타데이터 조건 검색 도구 (1차 목표)

- "예산 5억 이상", "OO부 발주", "3월 마감" 처럼 조건으로 공고 목록을 찾는 도구입니다.
- 벡터 검색·LLM 없이 공고 목록(벡터DB 메타데이터, summarize_rfp.load_catalog)을 pandas 로 필터링합니다.
- 조건: 발주기관(일부 일치), 사업명 키워드(일부 일치), 사업 금액 범위(원), 입찰 마감일 범위(YYYY-MM-DD)
  + 정렬(마감일/금액)·개수 제한. 결과는 사업명·발주기관·금액·마감일 표입니다.
- 결과 doc_id 목록을 artifact 로 돌려줘서, 1건이면 track_doc 이 대화 중인 공고로 정합니다 (후속 질문 대화).
- 메타데이터 키 이름은 ingest/metadata.py 의 METADATA_COLUMNS 와 같습니다.
"""

import pandas as pd
from langchain_core.tools import tool

from rfp_agent.agent.tools.summarize_rfp import _compact, load_catalog
from rfp_agent.config import load_config

AGENCY, TITLE, BUDGET, DEADLINE = "발주 기관", "사업명", "사업 금액", "입찰 참여 마감일"
SORT_COLUMNS = {"deadline": DEADLINE, "budget": BUDGET}
DEFAULT_LIMIT = 10
MAX_LIMIT = 30  # 도구 결과가 LLM 컨텍스트(Ollama 8k)를 너무 차지하지 않게


def _to_frame(catalog: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(catalog, columns=["doc_id", AGENCY, TITLE, BUDGET, DEADLINE])
    df[BUDGET] = pd.to_numeric(df[BUDGET], errors="coerce")
    df.loc[df[BUDGET] <= 0, BUDGET] = None  # 금액 0 은 미상으로 취급
    df["_deadline"] = pd.to_datetime(df[DEADLINE], errors="coerce")
    return df


def _contains(series: pd.Series, text: str) -> pd.Series:
    query = _compact(text)
    return series.fillna("").map(lambda value: query in _compact(str(value)))


def _parse_date(value: str | None, end_of_day: bool = False):
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"날짜 형식을 알 수 없습니다: {value!r} (YYYY-MM-DD 로 주세요)")
    if end_of_day and parsed == parsed.normalize():
        parsed += pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # 그날 하루 전체 포함
    return parsed


def filter_rfps(
    catalog: list[dict],
    *,
    agency: str | None = None,
    keyword: str | None = None,
    min_budget: int | None = None,
    max_budget: int | None = None,
    deadline_from: str | None = None,
    deadline_to: str | None = None,
    sort_by: str = "deadline",
    descending: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[dict], int]:
    """조건에 맞는 공고 (정렬·limit 적용 목록, 전체 일치 건수) 를 돌려줍니다."""
    df = _to_frame(catalog)
    mask = pd.Series(True, index=df.index)
    if agency:
        mask &= _contains(df[AGENCY], agency)
    if keyword:
        mask &= _contains(df[TITLE], keyword)
    # LLM 이 안 쓰는 조건에 0·빈 문자열을 넣는 경우가 있어 0 은 조건 없음으로 봅니다.
    if min_budget:
        mask &= df[BUDGET] >= min_budget
    if max_budget:
        mask &= df[BUDGET] <= max_budget
    start, end = _parse_date(deadline_from), _parse_date(deadline_to, end_of_day=True)
    if start is not None:
        mask &= df["_deadline"] >= start
    if end is not None:
        mask &= df["_deadline"] <= end

    matched = df[mask]
    column = "_deadline" if SORT_COLUMNS.get(sort_by, DEADLINE) == DEADLINE else BUDGET
    matched = matched.sort_values(column, ascending=not descending, na_position="last")
    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    rows = matched.head(limit).drop(columns="_deadline")
    records = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in rows.to_dict("records")]
    return records, len(matched)


def _budget_text(value) -> str:
    return f"{int(value):,}원" if value else "미상"


def format_results(records: list[dict], total: int, conditions: str) -> str:
    if not records:
        return f"조건({conditions})에 맞는 공고가 없습니다. 조건을 넓혀 보세요."
    count = f"전체 {total}건 중 {len(records)}건"
    # gemma 가 건수 줄을 빼먹어서, 답변에 그대로 옮기라는 지시를 결과에 함께 넣습니다.
    lines = [f"조건({conditions})에 맞는 공고: {count}",
             f"(답변 첫 줄에 '{count}'을 반드시 그대로 쓰고, 그 아래에 표를 옮기세요.)",
             "", "| # | 사업명 | 발주기관 | 사업 금액 | 입찰 마감일 |", "|---|---|---|---|---|"]
    for i, r in enumerate(records, start=1):
        lines.append(f"| {i} | {r[TITLE] or r['doc_id']} | {r[AGENCY] or '미상'} | "
                     f"{_budget_text(r[BUDGET])} | {r[DEADLINE] or '미상'} |")
    return "\n".join(lines)


def _conditions(**kwargs) -> str:
    labels = {"agency": "발주기관", "keyword": "사업명", "min_budget": "금액 ≥", "max_budget": "금액 ≤",
              "deadline_from": "마감 ≥", "deadline_to": "마감 ≤"}
    parts = []
    for key, label in labels.items():
        value = kwargs.get(key)
        if not value:
            continue
        parts.append(f"{label} {_budget_text(value) if 'budget' in key else value}")
    return ", ".join(parts) or "조건 없음"


def _filter(config: dict, **kwargs) -> tuple[str, dict]:
    try:
        records, total = filter_rfps(load_catalog(config), **kwargs)
    except ValueError as error:
        return str(error), {}
    conditions = _conditions(**kwargs)
    return format_results(records, total, conditions), {"doc_ids": [r["doc_id"] for r in records]}


_DESCRIPTION = (
    "발주기관, 사업명 키워드, 사업 금액 범위, 입찰 마감일 범위 같은 조건으로 RFP 공고 목록을 찾습니다. "
    "'OO기관 공고 목록', '예산 5억 이상 사업', '2024년 4월 마감 공고', '가장 비싼 사업 3개' 처럼 "
    "여러 공고를 조건으로 찾거나 세는 질문에 사용하세요. 문서 본문 내용 질문에는 search_rfp 를 쓰세요. "
    "금액은 원 단위 정수(5억 = 500000000), 날짜는 YYYY-MM-DD 입니다. "
    "sort_by 는 deadline(마감일) 또는 budget(금액), descending=true 면 큰 값·늦은 날짜부터, limit 는 최대 30건입니다."
)


def _tool_args(agency, keyword, min_budget, max_budget, deadline_from, deadline_to, sort_by, descending, limit):
    return dict(agency=agency, keyword=keyword, min_budget=min_budget, max_budget=max_budget,
                deadline_from=deadline_from, deadline_to=deadline_to, sort_by=sort_by,
                descending=descending, limit=limit)


@tool(description=_DESCRIPTION)
def filter_rfp_by_metadata(
    agency: str | None = None,
    keyword: str | None = None,
    min_budget: int | None = None,
    max_budget: int | None = None,
    deadline_from: str | None = None,
    deadline_to: str | None = None,
    sort_by: str = "deadline",
    descending: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> str:
    return _filter(load_config(), **_tool_args(
        agency, keyword, min_budget, max_budget, deadline_from, deadline_to, sort_by, descending, limit))[0]


def make_filter_rfp_by_metadata(config: dict):
    """에이전트가 쓰는 DB 설정을 조건 검색 도구에도 전달합니다."""

    @tool("filter_rfp_by_metadata", description=_DESCRIPTION, response_format="content_and_artifact")
    def configured_filter(
        agency: str | None = None,
        keyword: str | None = None,
        min_budget: int | None = None,
        max_budget: int | None = None,
        deadline_from: str | None = None,
        deadline_to: str | None = None,
        sort_by: str = "deadline",
        descending: bool = False,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[str, dict]:
        # 결과가 1건이면 track_doc 이 그 공고를 대화 중인 공고로 정합니다.
        return _filter(config, **_tool_args(
            agency, keyword, min_budget, max_budget, deadline_from, deadline_to, sort_by, descending, limit))

    return configured_filter
