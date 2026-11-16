# 유영관 RFP RAG 구현 및 로컬 웹 파일럿

선정한 네 노트북과 실제 구현인 `src/rfp_rag/`를 공유합니다.
개인 프로젝트의 Phase 1–8 기준선에서 실행에 필요한 Python 모듈을 추렸으며,
기존 생성 모델 `gpt-5.4-mini`와 BM25 + 문자 TF-IDF/RRF 검색을 유지했습니다.
SBERT/의미 검색 추가와 생성 모델 비교는 후속 Task 2입니다.

## 1. 포함 범위와 실행 순서

| 노트북 | 실행 내용 | Python 구현 |
| --- | --- | --- |
| `notebooks/04_younggwan_phase3_retrieval.ipynb` | 원문 JSONL → 청킹·색인 → 선택 문서 검색 | `workspace.py`, `retrieval/`, `workflows.search` |
| `notebooks/12_younggwan_phase5_final.ipynb` | 문서 전체의 11항목 요약·원문 검토표 | `summary_full/`, `summary_content/`, `workflows.summarize` |
| `notebooks/13_younggwan_phase6_conversation.ipynb` | 근거 답변·문서 비교·후속 질문 | `generation/`, `conversation/`, `workflows.submit_questions` |
| `notebooks/15_younggwan_phase8_pilot.ipynb` | 파일럿 6개 사례 검증·검토 상태 확인 | `pilot/validation.py`, `workflows.validate_once` |

파일명은 선정한 원본 노트북에 맞췄고, 내용은 공통 `.py`를 호출하는 실행 예제로 정리했습니다.
중복 셀·과거 패치 코드를 실행 경로에 복사하지 않았습니다. 실행 출력은 비웠고 `RUN_LIVE=False`가 기본입니다.
같은 커널·배치의 QA와 검증 셀 재실행은 저장된 요청/결과를 재사용합니다.
새로 실행하려면 배치 ID를 바꾸세요. 커널/CLI를 새로 시작하면 별도 예산이 시작됩니다.

## 2. 설치와 가상 문서 실행

팀 저장소 루트에서 아래 기여 폴더로 이동합니다. Python 3.11 또는 3.12와 uv가 필요합니다.

```bash
cd contributions/younggwan_rag
uv sync --extra notebook --extra openai
uv run python -m unittest discover -s tests -t tests -q
uv run rfp-rag build --demo --workspace workspaces/demo --config configs/rag.yaml
uv run rfp-rag catalog --workspace workspaces/demo
uv run rfp-rag search --workspace workspaces/demo --doc-id DEMO_A --question "사업 예산과 기간은?"
uv run rfp-rag ask --workspace workspaces/demo --doc-id DEMO_A --question "사업예산은?" --question "그 기간에 안정화도 포함돼?"
uv run rfp-rag summarize --workspace workspaces/demo --doc-id DEMO_A
uv run rfp-rag validate --workspace workspaces/demo --doc-id DEMO_A
```

위 명령은 실제 API를 호출하지 않습니다. 기본 QA는 원문 검색 후보를 보여 주며,
전체 요약은 API 미실행·미처리 상태를 명시합니다. 실제 생성 완료나 의미 정확도 통과로 해석하지 않습니다.
가상 문서의 ID는 `DEMO_A`, `DEMO_B`이며 실제 RFP 성능 평가용 자료가 아닙니다.
VS Code/Jupyter 커널은 이 폴더의 `.venv`를 선택합니다.

## 3. 이전과 같은 웹 화면 실행

```bash
# 위 build --demo를 먼저 실행합니다.
uv run rfp-rag serve --workspace workspaces/demo --port 8765
```

브라우저에서 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)를 엽니다.
문서 선택 → 선택 적용 → 질문 제출 순서로 원문 근거 검색을 확인합니다.
이 주소는 서버를 실행한 자신의 컴퓨터에서 접속하는 로컬 주소입니다.
종료는 서버 터미널에서 Ctrl+C를 누릅니다.

### 실제 답변 생성 켜기

```bash
# 이 폴더에서 최초 한 번만 복사합니다. 기존 키 파일을 덮어쓰지 마세요.
cp -n .env.example .env
# .env를 편집하여 OPENAI_API_KEY를 입력합니다.
uv run --env-file .env rfp-rag serve --workspace workspaces/demo --port 8765
```

웹 왼쪽의 **답변 생성 설정**을 열고 **실제 API 답변 생성**을 체크한 뒤,
**선택 문서 API 설정 적용**을 누르고 질문을 제출합니다.
`retrieved / disabled`이면 원문 검색 모드입니다. 생성이 켜진 상태에서만 선택 문서 근거를 API로 전송합니다.
모델 호출은 사용자 제출 시 수행하며, 문서 허용 목록·호출 수·입력량 상한을 적용합니다.
키를 노트북 셀, Git, 화면 캡처에 넣지 않습니다.

CLI에서 실제 답변을 생성하려면 다음처럼 실행합니다.

```bash
uv run rfp-rag ask --workspace workspaces/demo --doc-id DEMO_A --question "완료·검수 조건은?" --live --env-file .env
```

## 4. 실제 문서 연결

문서와 색인은 Git에 포함하지 않습니다. 각 팀원이 원문 블록 JSONL을 별도로 받아 구축합니다.
기존 개인 프로젝트의 `rfp_phase5_local_20261111/data/source_documents.jsonl`을 사용할 수 있습니다.

```bash
uv run rfp-rag build --source /absolute/path/source_documents.jsonl --workspace workspaces/team-v1 --config configs/rag.yaml
uv run rfp-rag catalog --workspace workspaces/team-v1
uv run --env-file .env rfp-rag serve --workspace workspaces/team-v1 --port 8765
```

`--source`는 원문 파일/PDF 자체가 아닌 파싱 결과 JSONL 경로입니다.
JSONL의 각 행에는 `doc_id`, `canonical_doc_id`, 원본 파일 `sha256`, `file_name`, `detected_format`,
`blocks`가 필요합니다. 각 블록에는 `block_id`, `text`, 실제 `locator`와 `page` 또는 `section`을 보존합니다.
사본은 `canonical_doc_id`와 `is_duplicate_alias`로 원본에 연결합니다.
표는 위치가 있는 원문 `blocks`에 포함되어야 하며 `tables`만 있으면 검색 본문에 들어가지 않습니다.
없는 페이지·위치나 SHA를 임의로 채우지 않습니다. 검토되지 않은 OCR·참조용 중복 뷰는 제외합니다.
이번 기여에는 PDF/HWP 파서를 추가하지 않았습니다. 팀 파싱 결과와 이 입력 계약을 맞추는 작업이 필요합니다.

새 workspace를 구축할 때 현재 환경에서 청크와 인덱스를 생성합니다. OpenAI 임베딩 API를 호출하지 않습니다.
원문·코드·설정·라이브러리가 바뀌면 기존 workspace를 덮어쓰지 않고 새 경로를 사용합니다.
설정은 `configs/rag.yaml`이며, 실행 예산·생성 모델을 바꾸면 새 workspace를 만듭니다.

## 5. Phase 1–8 개발 내역

| 단계 | 개발한 내용 | 이번 공유 범위 |
| --- | --- | --- |
| Phase 1 | 기능 범위, 수용 기준, 출처·평가 계약 | 실행·검토 기준으로 반영 |
| Phase 2 | PDF/HWP/HWPX 원문 추출과 위치·원본 SHA·사본 추적 | 파싱 결과 입력 계약; 파서·실제 원문은 제외 |
| Phase 3 | 원문 위치를 유지하는 청킹, BM25·문자 TF-IDF/RRF 검색 | `retrieval/`, `workspace.py` |
| Phase 4 | 구조화된 근거 답변, 원문 인용·문서 범위 검사 | `generation/` |
| Phase 5 | 문서 전체 11항목 요약, 숫자·조건 취합, 실패/재개와 검토 기록 | `summary_fields/`, `summary_full/`, `summary_content/`, `recovery/` |
| Phase 6 | 문서 선택·전환·비교·후속 질문·모호한 참조 처리 | `conversation/` |
| Phase 7 | 검색·범위·인용·지연·사용량 검사와 사람 검토 분리 | `evaluation/`, 회귀 검사 |
| Phase 8 | 로컬 웹 파일럿, 실행 기록, 주장별 인용 검토 신호, 6개 검증 사례 | `pilot/`, CLI, 노트북 15 |

주요 보완 사항:

- 모델이 긴 원문을 임의로 합쳐 인용하지 않도록 원문 연결을 검사합니다.
- 구체적 범위 질문은 ASCII 인용 참조 ID를 사용하고 로컬에서 정확한 원문으로 확장합니다.
- 실제 API에서 발생했던 `invalid_json_schema` 문제를 보완한 전송 형식을 유지합니다.
- 각 주장에 붙은 인용의 명칭·요구사항 ID 누락을 검토 신호로 표시합니다.
- 이 신호는 의미적 정답 판정이 아닙니다. 숫자·조건·부정문과 주장 전체의 근거는 사람이 대조해야 합니다.

## 6. 검증과 남은 일

기준 패키지는 579개 unittest 실행에서 실패 0, 실제 저장 코퍼스가 필요한 검사 1개 생략이었습니다.
네 노트북의 Python 셀을 각각 두 번 실행하고 wheel의 독립 실행 경로를 확인했으며 실제 API 호출은 0회였습니다.
과거 검증 환경에서 PyPI가 차단되어 새 가상환경 `uv sync`는 확인하지 못했습니다.
업로드 스크립트는 실행자의 환경에서 `uv sync`·회귀 검사·CLI·로컬 HTTP 검사를 통과하고,
생성된 `uv.lock`을 포함한 경우에만 commit/push합니다. 이번 실행 결과는 `PUBLISH_VALIDATION.json`에 기록합니다.
실제 Jupyter 화면·브라우저 렌더링과 실제 문서 답변의 의미 정확도는 별도 확인 대상입니다.

`semantic_accuracy=null`, `review_pending`은 의미 검토가 아직 완료되지 않았다는 뜻입니다.
기계 검사 통과를 답변 정답률이나 전체 실제 문서 요약 완료로 표시하지 않습니다.
실행 기록·검토표는 workspace 아래 `phase*_outputs/`에 저장되며 Git에서는 제외합니다.

팀 통합과 후속 개선:

1. 팀 `ParsedDocument`와 원문 블록·위치 입력 계약 연결.
2. `workflows.search`/`adapters.to_langchain_documents`를 팀 `retrieve(query, config, filters)`와 연결.
3. 기존 생성 클라이언트를 팀 `get_llm(config)`/Ollama 경계와 연결.
4. 팀 LangGraph `build_graph(config)`와 API `AskResponse(answer, sources)` 연결.
5. 동일한 질문·근거·프롬프트로 생성 모델만 비교한 뒤, 생성 모델을 고정하고 의미 검색 결합을 비교.
6. 개선이 확인된 조합으로 새 문서·질문을 독립 평가.

이 폴더는 독립 실행 가능한 구현 기여입니다. 팀 그래프/API에 이미 연결된 상태는 아닙니다.
팀 루트의 `uv sync` 대신 이 기여 폴더에서 `uv sync`를 실행합니다.
기존 팀 코드와 공통 인터페이스를 바꾸는 통합은 별도 작업으로 진행합니다.

## 7. 출처와 Git 포함 정책

개인 기준선 `f99e7942`를 정리한 공유 패키지 커밋:
`TheMostEvident/kor-rfp-rag-agent-openai@cd1b7ccf7f5ee09f3fc6ff4c20c4314173642eeb`.
모듈별 이동 경로는 `SOURCE_MAP.json`에 기록되어 있습니다.
원본 04/12/13/15에 대응하는 네 실행 노트북, 실행 `.py`, 설정, 의존성·잠금 파일, 테스트와 설명 문서를 포함합니다.
중간 패치·중복 수정본·평가 정답·실제 원문·색인·캐시·실행 로그·사람 검토 기록·키 파일은 포함하지 않습니다.
