"""[평가·데모] 평가 지표

- 평가셋(eval/questions.jsonl)의 정답과 에이전트 답변을 비교하는 지표 함수를 작성합니다.
- 지표는 팀이 직접 정해야 하고, 왜 그 지표를 골랐는지 노션 「의사결정 로그」(분류: 평가)에 남겨 주세요.
- 지표 아이디어:
  - 검색: Hit@k (정답 문서가 검색 결과 top_k 안에 있는지), MRR
  - 답변: 정답 키워드 포함 여부, LLM-as-a-judge 점수(정확성·근거 충실도), 응답 시간, 토큰 비용
  - 라이브러리: RAGAS 같은 평가 도구도 비교해 볼 수 있음 (`uv add ragas`)
"""


def hit_at_k(retrieved_doc_ids: list[str], answer_doc_id: str) -> float:
    """정답 문서가 검색 결과 안에 있으면 1.0, 없으면 0.0"""
    return 1.0 if answer_doc_id in retrieved_doc_ids else 0.0


# TODO: 답변 품질 지표 추가 (예: def llm_judge(question, answer, reference) -> float)
