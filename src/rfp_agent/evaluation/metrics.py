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


def score_unanswerable(answer: str) -> float:
    """문서에 없는 정보 질문(unanswerable)에 대해 환각 없이 미기재·확인 불가를 답했는지 채점합니다."""
    if not answer or not answer.strip():
        return 0.0
    abstention_cues = (
        "확인할 수 없",
        "명시되어 있지 않",
        "기재되어 있지 않",
        "찾을 수 없",
        "포함되어 있지 않",
        "정해진 바 없",
        "알 수 없",
        "제공된 문서",
        "제공된 제안요청서",
        "과업지시서만으로는",
        "없습니다",
        "없다",
    )
    return 1.0 if any(cue in answer for cue in abstention_cues) else 0.0


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
) -> dict[str, float]:
    """LLM-as-a-judge 채점: 정확성(accuracy)과 근거 충실도(faithfulness)를 0.0~1.0으로 평가합니다."""
    if llm is None:
        from rfp_agent.config import load_config
        from rfp_agent.llm import get_llm

        llm = get_llm(load_config())

    context_block = f"\n[검색된 근거 문서]\n{context}\n" if context else ""
    prompt = (
        "당신은 공공 입찰공고 RAG 시스템의 답변 채점관입니다.\n"
        "아래 기준에 따라 모델 답변을 0.0에서 1.0 사이 점수로 평가하고 반드시 JSON 한 줄로만 답하세요.\n"
        "- accuracy (정확성): 정답(Reference)의 핵심 사실·금액·날짜·조건이 빠짐없이 정확히 일치하는가\n"
        "- faithfulness (근거 충실도): 정답/근거에 없는 내용을 임의로 지어내지(환각) 않았는가\n"
        '출력 형식 예시: {"accuracy": 0.9, "faithfulness": 1.0}\n\n'
        f"[질문] {question}\n"
        f"[정답] {reference}\n"
        f"{context_block}"
        f"[모델 답변] {answer}"
    )
    try:
        resp = llm.invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\{.*?\}", text, re.S)
        if match:
            data = json.loads(match.group(0))
            acc = round(max(0.0, min(1.0, float(data.get("accuracy", 0.0)))), 4)
            faith = round(max(0.0, min(1.0, float(data.get("faithfulness", acc)))), 4)
            avg = round((acc + faith) / 2, 4)
            return {"accuracy": acc, "faithfulness": faith, "score": avg}
    except Exception:
        pass
    return {"accuracy": 0.0, "faithfulness": 0.0, "score": 0.0}


def estimate_token_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """모델별 1M 토큰 단가 기준 예상 API 비용(USD)을 계산합니다 (로컬 Ollama/Gemma는 0.0)."""
    model_lower = model.lower()
    if "gemma" in model_lower or "ollama" in model_lower or "local" in model_lower:
        return 0.0

    # 1M(1,000,000) 토큰당 USD 단가 (input, output)
    pricing: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1": (2.00, 8.00),
    }
    in_rate, out_rate = pricing.get("gpt-4o-mini", (0.15, 0.60))
    for key, rates in pricing.items():
        if key in model_lower:
            in_rate, out_rate = rates
            break

    cost = (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate
    return round(cost, 6)
