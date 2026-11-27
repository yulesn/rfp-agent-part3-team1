"""[프론트·백엔드·평가] 평가 지표

- 평가셋(eval/questions.jsonl)의 정답과 에이전트 답변을 비교하는 지표 함수를 작성합니다.
- 지표는 팀이 직접 정해야 하고, 왜 그 지표를 골랐는지 노션 「의사결정 로그」(분류: 평가)에 남겨 주세요.
- 지표 아이디어:
  - 검색: Hit@k (정답 문서가 검색 결과 top_k 안에 있는지), MRR
  - 답변: 정답 키워드 포함 여부, LLM-as-a-judge 점수(정확성·근거 충실도), 응답 시간, 토큰 비용
  - 라이브러리: RAGAS 같은 평가 도구도 비교해 볼 수 있음 (`uv add ragas`)
"""


from __future__ import annotations

import json
import re
from typing import Any


def _to_doc_list(answer_doc_id: str | list[str] | None) -> list[str]:
    """단일 doc_id(str) 또는 다중 doc_id(list[str]) 입력을 리스트로 통일합니다."""
    if answer_doc_id is None:
        return []
    if isinstance(answer_doc_id, str):
        return [answer_doc_id] if answer_doc_id.strip() else []
    return [doc_id for doc_id in answer_doc_id if doc_id]


def hit_at_k(retrieved_doc_ids: list[str], answer_doc_id: str | list[str]) -> float | None:
    """정답 문서가 검색 결과 안에 있으면 1.0, 없으면 0.0 (정답 문서가 없는 질문은 None)."""
    targets = _to_doc_list(answer_doc_id)
    if not targets:
        return None
    retrieved_set = set(retrieved_doc_ids)
    return 1.0 if any(doc_id in retrieved_set for doc_id in targets) else 0.0


def mrr(retrieved_doc_ids: list[str], answer_doc_id: str | list[str]) -> float | None:
    """Mean Reciprocal Rank: 검색 결과에서 첫 번째 정답 문서가 등장한 순위의 역수(1/rank)."""
    targets = _to_doc_list(answer_doc_id)
    if not targets:
        return None
    target_set = set(targets)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in target_set:
            return round(1.0 / rank, 4)
    return 0.0


def recall_at_k(retrieved_doc_ids: list[str], answer_doc_id: str | list[str]) -> float | None:
    """정답 문서 집합 대비 검색된 정답 문서 비율 (multi_doc 비교 질문 대응)."""
    targets = _to_doc_list(answer_doc_id)
    if not targets:
        return None
    target_set = set(targets)
    retrieved_set = set(retrieved_doc_ids)
    return round(len(target_set & retrieved_set) / len(target_set), 4)


def _normalize_numbers(text: str) -> set[str]:
    """텍스트에서 숫자를 추출하고 천원/백만원/억원 단위를 원 단위로 정규화합니다."""
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text)
    nums = set(re.findall(r"\d+(?:\.\d+)?", cleaned))

    for val in re.findall(r"(\d+(?:\.\d+)?)\s*천원", cleaned):
        nums.add(str(int(float(val) * 1_000)))
    for val in re.findall(r"(\d+(?:\.\d+)?)\s*백만원", cleaned):
        nums.add(str(int(float(val) * 1_000_000)))
    for val in re.findall(r"(\d+(?:\.\d+)?)\s*억", cleaned):
        nums.add(str(int(float(val) * 100_000_000)))

    return nums


def _extract_keywords(text: str) -> set[str]:
    """정답 비교를 위한 핵심 키워드(2글자 이상 한글·영문·숫자)를 추출합니다."""
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    stopwords = {
        "입니다", "이다", "합니다", "한다", "있으며", "있습니다", "있다", "대한", "통해",
        "위한", "따른", "경우", "이상", "이하", "이내", "부터", "까지", "제공된", "제안요청서",
        "사업", "용역", "구축", "운영", "시스템", "기준", "관련", "및", "등",
    }
    tokens = set(re.findall(r"[가-힣a-z0-9]{2,}", cleaned))
    return {t for t in tokens if t not in stopwords}


def keyword_match_score(answer: str, reference: str) -> float:
    """정답 키워드 및 핵심 수치 포함 비율을 0.0~1.0 점수로 계산합니다."""
    if not answer or not answer.strip():
        return 0.0

    ref_nums = _normalize_numbers(reference)
    ans_nums = _normalize_numbers(answer)
    ref_keywords = _extract_keywords(reference)
    ans_keywords = _extract_keywords(answer)

    kw_recall = len(ref_keywords & ans_keywords) / len(ref_keywords) if ref_keywords else 1.0

    if ref_nums:
        num_recall = len(ref_nums & ans_nums) / len(ref_nums)
        return round(0.6 * num_recall + 0.4 * kw_recall, 4)
    return round(kw_recall, 4)


# "문서에 없다"고 답했는지 보는 표현. "제공된 문서", "없습니다" 같은 흔한 말은 넣지 않습니다
# (엉뚱한 문서를 보고 답해도 통과해 버림). "명시되어 있지는 않습니다"처럼 조사가 붙은 것도 잡습니다.
ABSTENTION_PATTERNS = (
    r"(명시|기재|포함|언급|규정|제시|나와|정해져|기술)\s*(되어|돼)?\s*있지\s*(는|도)?\s*않",
    r"(명시|기재|언급|확인|규정)되지\s*(는|도)?\s*않",
    r"(확인할|파악할|찾을|알)\s*수\s*(가\s*)?없",
    r"(명시|기재|정보|내용)\s*없음",
    r"정해진\s*바\s*(가\s*)?없",
)


def score_unanswerable(answer: str) -> float:
    """문서에 없는 정보 질문(unanswerable)에 대해 환각 없이 미기재·확인 불가를 답했는지 채점합니다."""
    if not answer or not answer.strip():
        return 0.0
    return 1.0 if any(re.search(p, answer) for p in ABSTENTION_PATTERNS) else 0.0


def _first_call(calls: list[tuple[str, list[str]]], tool_name: str) -> list[str]:
    """tool_name 의 첫 호출 결과. 그 도구를 안 썼으면 문서를 1개 이상 돌려준 첫 도구 호출 결과를 씁니다.

    (예: filter_rfp_by_metadata 를 0건 → 1건 순서로 불렀으면 1건짜리를 봄)
    """
    for name, doc_ids in calls:
        if name == tool_name:
            return doc_ids
    return next((doc_ids for _, doc_ids in calls if doc_ids), [])


def retrieval_scores(
    q_type: str,
    calls: list[tuple[str, list[str]]],
    answer_doc_id: str | list[str],
    k: int = 5,
) -> dict[str, float | None]:
    """검색 한 번의 top-k 기준 hit@k, MRR, recall@k.

    calls: 에이전트의 도구 호출 순서대로 (도구 이름, 검색된 doc_id 목록).
    - summary: 첫 summarize_rfp 호출이 고른 공고가 맞는지
    - multi_doc: 정답 문서마다 어느 한 번의 search_rfp top-k 에 들어왔는지 본다
      hit 은 정답 문서를 전부 찾았으면 1 (비교 질문에 답할 수 있는지), recall 은 찾은 비율,
      MRR 은 정답 문서마다 가장 좋은 순위의 역수를 평균
    - 그 외: 첫 search_rfp 호출의 top-k
    정답 문서가 없는 질문은 모두 None.
    """
    targets = _to_doc_list(answer_doc_id)
    if not targets:
        return {"hit_at_k": None, "mrr": None, "recall_at_k": None}

    if q_type == "multi_doc":
        searches = [ids[:k] for name, ids in calls if name == "search_rfp"] or [ids[:k] for _, ids in calls]
        found, rrs = [], []
        for target in targets:
            ranks = [ids.index(target) + 1 for ids in searches if target in ids]
            found.append(1.0 if ranks else 0.0)
            rrs.append(1.0 / min(ranks) if ranks else 0.0)
        return {
            "hit_at_k": 1.0 if all(found) else 0.0,
            "mrr": round(sum(rrs) / len(targets), 4),
            "recall_at_k": round(sum(found) / len(targets), 4),
        }

    tool_name = "summarize_rfp" if q_type == "summary" else "search_rfp"
    ranked = _first_call(calls, tool_name)[:k]
    return {
        "hit_at_k": hit_at_k(ranked, targets),
        "mrr": mrr(ranked, targets),
        "recall_at_k": recall_at_k(ranked, targets),
    }


def score_summary(answer_facts: dict[str, Any], answer: str) -> tuple[float, dict[str, float]]:
    """summary 문항의 11개 공통 항목(answer_facts)별 핵심 정보 포함률을 채점합니다."""
    if not answer_facts:
        return 0.0, {}
    if not answer or not answer.strip():
        return 0.0, {k: 0.0 for k in answer_facts}

    missing_cues = ("명시되지 않", "미기재", "공고문 참조", "확인할 수 없", "별도")
    field_scores: dict[str, float] = {}

    for key, val in answer_facts.items():
        ref_str = " ".join(str(x) for x in val) if isinstance(val, list) else str(val or "")
        if any(cue in ref_str for cue in ("명시되지 않", "미기재", "공고문 참조")):
            matched = any(cue in answer for cue in missing_cues)
            field_scores[key] = 1.0 if matched else 0.0
        else:
            field_scores[key] = keyword_match_score(answer, ref_str)

    avg_score = round(sum(field_scores.values()) / len(field_scores), 4)
    return avg_score, field_scores


def score_answer(item: dict[str, Any], answer: str) -> dict[str, Any]:
    """질문 유형(type)에 맞춰 답변 점수와 세부 내역을 반환합니다."""
    q_type = item.get("type", "single_doc")

    if q_type == "summary":
        facts = item.get("answer_facts", {})
        score, per_field = score_summary(facts, answer)
        matched_cnt = sum(1 for v in per_field.values() if v >= 0.5)
        return {
            "answer_score": score,
            "detail": f"11개 항목 중 {matched_cnt}/{len(per_field)}개 충족",
        }

    if q_type == "unanswerable":
        score = score_unanswerable(answer)
        return {
            "answer_score": score,
            "detail": "환각 억제 성공" if score == 1.0 else "환각 의심(미기재 표현 누락)",
        }

    reference = str(item.get("answer", ""))
    score = keyword_match_score(answer, reference)
    return {
        "answer_score": score,
        "detail": f"유형={q_type}",
    }

def llm_judge(
    question: str,
    answer: str,
    reference: str,
    context: str = "",
    llm: Any = None,
    q_type: str = "",
) -> dict[str, Any]:
    """LLM-as-a-judge 채점: reason을 먼저 작성한 뒤 accuracy와 faithfulness를 0.0~1.0으로 평가합니다.

    summary 문항은 항목이 여러 개라, 하나만 빠져도 0점이 되지 않도록 맞힌 항목 비율로 accuracy 를 줍니다.
    """
    if llm is None:
        from rfp_agent.config import load_config
        from rfp_agent.llm import get_llm

        llm = get_llm(load_config())

    context_block = context if context.strip() else "(검색된 근거 없음)"
    if q_type == "summary":
        accuracy_rule = (
            "   - [정답(Reference)]은 항목별 정답(JSON)입니다. 항목마다 모델 답변이 정답의 핵심 사실·수치를 맞게 썼는지 보고,\n"
            "     맞힌 항목 수 / 전체 항목 수를 accuracy 로 주세요. 정답이 있는데 '문서에 명시 없음'이라고 쓴 항목은 틀린 것으로 셉니다.\n"
            "     reason 에는 틀리거나 빠진 항목 이름을 적으세요.\n"
        )
    else:
        accuracy_rule = (
            "   - 정답에 구체적인 사실·수치·명칭이 있는데 모델이 '문서에 명시 없음', '확인할 수 없음' 등으로 답하거나 핵심 사실을 누락·왜곡하면 accuracy는 반드시 0.0입니다.\n"
        )
    prompt = (
        "당신은 공공 입찰공고 RAG 시스템의 엄격한 답변 채점관입니다.\n"
        "아래 규칙에 따라 모델 답변을 평가하고, 반드시 JSON 한 줄로만 출력하세요.\n\n"
        "[채점 규칙]\n"
        "1. reason: 채점 근거를 한국어 한 줄로 먼저 작성하세요.\n"
        "2. accuracy (정확성, 0.0~1.0): 오직 [질문]과 [정답(Reference)]만 비교하세요. [검색된 근거 문서]는 절대 참고하지 마세요.\n"
        f"{accuracy_rule}"
        "3. faithfulness (근거 충실도, 0.0~1.0): 오직 [검색된 근거 문서]만 기준으로, 모델 답변이 근거에 없는 내용을 임의로 지어내지(환각) 않았는지 평가하세요.\n\n"
        '출력 형식 예시: {"reason": "정답은 KCMVP이나 모델이 문서에 명시 없음이라고 오답을 냄", "accuracy": 0.0, "faithfulness": 1.0}\n\n'
        f"[질문] {question}\n"
        f"[정답(Reference)] {reference}\n"
        f"[검색된 근거 문서]\n{context_block}\n"
        f"[모델 답변] {answer}"
    )
    try:
        resp = llm.invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\{.*?\}", text, re.S)
        if match:
            data = json.loads(match.group(0))
            reason = str(data.get("reason", "")).strip()
            acc = round(max(0.0, min(1.0, float(data.get("accuracy", 0.0)))), 4)
            faith = round(max(0.0, min(1.0, float(data.get("faithfulness", acc)))), 4)
            avg = round((acc + faith) / 2, 4)
            return {"reason": reason, "accuracy": acc, "faithfulness": faith, "score": avg}
    except Exception as exc:
        return {"reason": f"채점 파싱 오류: {exc}", "accuracy": 0.0, "faithfulness": 0.0, "score": 0.0}
    return {"reason": "JSON 응답 형식 오류", "accuracy": 0.0, "faithfulness": 0.0, "score": 0.0}

def sum_usage(usage_by_model: dict[str, dict[str, Any]]) -> tuple[int, int]:
    """get_usage_metadata_callback() 결과(모델별 사용량)를 합쳐 (입력 토큰, 출력 토큰)으로 돌려줍니다."""
    in_tok = sum(int(u.get("input_tokens", 0)) for u in usage_by_model.values())
    out_tok = sum(int(u.get("output_tokens", 0)) for u in usage_by_model.values())
    return in_tok, out_tok


def estimate_token_cost(input_tokens: int, output_tokens: int, price_per_1m: dict | None) -> float | None:
    """설정 파일의 price_per_1m(1M 토큰당 USD, input/output)으로 비용을 계산합니다.

    단가가 안 적혀 있으면 None (0 이 아니라 "계산 안 됨"). 로컬 모델은 yaml 에 0 을 적어 주세요.
    """
    if not price_per_1m:
        return None
    in_rate, out_rate = price_per_1m.get("input"), price_per_1m.get("output")
    if in_rate is None or out_rate is None:
        return None
    return round(input_tokens / 1_000_000 * float(in_rate) + output_tokens / 1_000_000 * float(out_rate), 6)
