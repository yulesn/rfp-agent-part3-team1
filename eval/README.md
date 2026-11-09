# eval/

- `questions.jsonl`: 평가용 질문·정답 세트. 한 줄에 질문 하나 (JSON).
  - `id`, `question`, `answer`(정답), `answer_doc_id`(정답 근거 문서), `type`(질문 유형)
  - 위 두 줄은 형식 예시입니다. 실제 RFP를 읽고 질문을 만들면서 지워 주세요.
  - 질문 유형을 골고루 섞어 주세요: 단일 문서 사실 확인, 여러 문서 비교, 조건 검색, 요약, 문서에 없는 내용(모른다고 답해야 함) 등
- `results/`: `scripts/evaluate.py` 실행 결과 (git에 올라가지 않음)
