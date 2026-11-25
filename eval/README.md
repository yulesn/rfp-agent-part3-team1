# eval/

- `questions.jsonl`: 평가용 질문·정답 세트. 한 줄에 질문 하나 (JSON).
  - `id`, `question`, `answer`(정답), `answer_doc_id`(정답 근거 문서), `type`(질문 유형)
  - 위 두 줄은 형식 예시입니다. 실제 질문을 넣으면서 지워 주세요.
  - 만드는 방식: AI로 질문·정답 초안을 생성한 뒤 사람이 원문과 대조해 검증 (검증 안 된 문항은 넣지 않기)
  - 질문 유형을 골고루 섞어 주세요: 단일 문서 사실 확인, 여러 문서 비교, 조건 검색, 요약, 문서에 없는 내용(모른다고 답해야 함) 등
- `results/`: `scripts/evaluate.py` 실행 결과 (git에 올라가지 않음)

# 평가 파이프라인 가이드 (`eval/`)

RFP RAG 에이전트의 검색 정확도와 답변 품질을 자동으로 측정하고 비교하기 위한 평가셋 및 실행 가이드입니다.

---

## 1. 폴더 및 주요 파일 구성

* **`eval/questions.jsonl`**: 검증 완료된 50개 평가 문항 데이터셋 (한 줄에 JSON 1개)
  * 포함 유형(7종): `single_doc`, `multi_doc`, `table_lookup`, `numeric_reasoning`, `paraphrase`, `summary`, `unanswerable`
* **`scripts/evaluate.py`**: 평가 실행 메인 스크립트
* **`src/rfp_agent/evaluation/metrics.py`**: 검색·답변·LLM Judge 채점 함수 모음
* **`eval/results/`**: 실행 결과 저장 폴더 (`.gitignore` 처리됨)
  * `{YYYYMMDD}_{HHMMSS}_{설정명}.csv`: 문항별 상세 채점표 및 모델 답변
  * `{YYYYMMDD}_{HHMMSS}_{설정명}_summary.json`: 전체 평균 및 유형별 요약 리포트

---

## 2. 실행 전 사전 준비

1. **문서 및 벡터 인덱스 준비**
   * `data/processed/documents.jsonl` 및 `data/vectorstore/`가 생성되어 있어야 합니다. (없을 경우 `uv run python scripts/build_index.py` 먼저 실행)
2. **모델 환경 확인**
   * **로컬 모델 (`configs/ollama.yaml`)**: Ollama 실행 상태 및 모델 다운로드 확인 (`ollama pull gemma4:12b`)
   * **OpenAI API / LLM Judge 사용 시**: 프로젝트 루트 `.env` 파일에 `OPENAI_API_KEY` 등록

---

## 3. 자주 쓰는 실행 명령어

빠른 동작 확인 (앞 3문항 스모크 테스트, API 비용 0원)
bash
uv run python scripts/evaluate.py --config configs/ollama.yaml --limit 3

50문항 전체 기준선(Baseline) 평가 (규칙 기반 채점만 실행)
Bash
uv run python scripts/evaluate.py --config configs/ollama.yaml

LLM Judge 채점 포함 (채점 모델은 설정 파일의 judge_llm 섹션으로 정함)
답변 생성은 llm, 채점은 judge_llm(예: openai gpt-5-mini)으로 분리해 셀프 채점 편향을 막습니다. 채점 모델을 바꾸려면 yaml 의 judge_llm 을 수정하세요.
Bash
uv run python scripts/evaluate.py --config configs/ollama.yaml --limit 5 --llm-judge