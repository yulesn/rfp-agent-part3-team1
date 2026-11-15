# 검색 베이스라인 실행 가이드

검색 담당 김원태. 청크 → OpenAI 임베딩 → 로컬 Chroma → `retrieve()` → 본문·출처 반환을 구현했습니다.
답변 생성 LLM과 임베딩 모델은 별개입니다. 이 검색 단계에는 Ollama 답변 생성 서버가 필요하지 않습니다.

## 환경과 입력

저장소 루트에서 Python 3.11 또는 3.12로 실행합니다.

```bash
uv sync --frozen
cp .env.example .env
# .env의 OPENAI_API_KEY에 팀 키 입력. 키는 코드·노트북 출력·Git에 넣지 않음.
uv run python scripts/retrieval_index.py doctor
```

`doctor`는 키 설정 여부와 의존성 로딩만 확인하며 API를 호출하지 않습니다.
기본 샘플 설정은 `configs/retrieval_sample.yaml`: 1000자/겹침 200자, `text-embedding-3-small`, top_k=5입니다.
샘플 DB는 `data/vectorstore_sample_openai_v1/`, 컬렉션은 `rfp_sample_v1`입니다.

### 이미 청크를 받은 경우

JSONL의 각 행은 다음 형태입니다. 본문과 값은 예시이며 metadata의 실제 키 이름은 전달받은 것을 보존합니다.

```json
{"page_content":"사업 예산은 1억원입니다.","metadata":{"doc_id":"사업.hwp","chunk_id":"사업.hwp#0","사업 금액":100000000}}
```

자동 검사: 빈 본문, doc_id/chunk_id 누락, `doc_id#순번` 형식, 중복 ID, 금액 정수, 날짜 YYYY-MM-DD.
본 베이스라인의 metadata 값은 문자열·숫자·불리언만 허용합니다. 빈 값은 키를 생략합니다.
원문과의 내용 일치, 파싱 누락, 페이지 정확성은 별도로 확인해야 합니다.

```bash
# 입력 파일 경로를 실제 받은 파일로 변경. API 호출 없음.
uv run python scripts/retrieval_index.py validate --chunks data/processed/chunks.jsonl --report data/processed/validation.json
```

이후 아래 build/search 명령에서 같은 파일을 사용합니다. JSONL 파일 형식은 수령용 CLI의 형식이며,
팀 파서가 메모리의 `list[Document]`를 넘기는 경우 `build_vectorstore(chunks, config)`를 직접 호출합니다.

### 청크 없이 CSV 텍스트로 확인하는 경우

제공 CSV를 `data/raw/data_list.csv`에 저장합니다. UTF-8 CSV이며 `파일명`, `텍스트` 열이 필요합니다.
학습용 고정 문자 수 분할로 유효한 첫 3문서를 준비합니다. 팀의 파싱·청킹 구현을 대체하지 않습니다.

```bash
uv run python scripts/retrieval_sample.py prepare --csv data/raw/data_list.csv
uv run python scripts/retrieval_index.py validate --chunks data/processed/retrieval_sample/chunks.jsonl --report data/processed/retrieval_sample/validation.json
```

이미 준비한 파일은 덮어쓰지 않습니다. 새로운 샘플은 설정의 processed_dir을 새 경로로 지정합니다.

## 실제 임베딩과 재접속·검색

아래는 샘플 경로입니다. 팀 청크를 받은 경우 `--chunks` 경로를 변경하세요.
build와 search는 실제 OpenAI API를 사용합니다. 문서와 질문을 같은 임베딩 모델로 변환합니다.

```bash
uv run python scripts/retrieval_index.py build --chunks data/processed/retrieval_sample/chunks.jsonl --report data/processed/retrieval_sample/build-report.json
# 새 프로세스에서 ready 상태와 청크 수 확인. inspect 자체는 API 호출 없음.
uv run python scripts/retrieval_index.py inspect
uv run python scripts/retrieval_sample.py search --query "이 사업의 예산과 기간은?"
```

팀 전체 DB를 만들 때는 명령의 **서브명령 앞에** `--config configs/default.yaml`을 지정합니다.
완성된 문서 처리 코드와 원본이 준비되면 기존 `scripts/build_index.py`로 전체 파이프라인을 실행할 수 있습니다.

```python
from rfp_agent.config import load_config
from rfp_agent.retrieval.retriever import retrieve

config = load_config("configs/retrieval_sample.yaml")
# 저장된 DB를 다시 열고 질문을 임베딩하여 관련 청크를 반환합니다.
documents = retrieve("사업 예산은 얼마인가?", config)
for document in documents:
    print(document.metadata["chunk_id"], document.page_content)
```

반환은 `list[Document]`이며 점수는 포함하지 않습니다. top_k는 YAML에서 읽습니다.
`filters`는 현재 Chroma where 문법을 전달하지만 공통 필터 계약과 조건 목록 검색은 후속 조율 대상입니다.
`search_rfp` 도구는 기본 설정을 읽어 본문·파일명·청크 ID·페이지(있을 때)를 반환합니다.
샘플 DB는 위 CLI 또는 설정을 넘긴 `retrieve()`로 확인하세요. 그래프 전체 연결은 에이전트 담당과 조율합니다.

## 저장·공유와 실패 처리

- 기존 컬렉션은 덮어쓰지 않습니다. 청크나 임베딩 모델이 바뀌면 새 경로/컬렉션으로 재생성합니다.
- DB 부재, 미완성 인덱스, 임베딩 모델 불일치는 오류입니다. 검색 조건 불일치는 빈 목록입니다.
- 컬렉션에는 모델과 완료 상태를 저장하며 API 키는 저장하지 않습니다.
- 팀 전체 DB는 김원태가 생성해 `data/vectorstore/` 폴더 전체를 Drive로 공유합니다.
- 공유 전 DB 사용 프로세스를 종료하고 DB 전체·사용 YAML·생성 보고서·uv.lock을 함께 전달합니다.
- `.env`, 원본 CSV, 청크, DB 파일, 실행 보고서는 Git 제외입니다. 받는 사람도 질문 임베딩용 키가 필요합니다.

## 확인 결과와 남은 범위

2026-11-15 실제 제공 텍스트 3문서·9청크로 임베딩 → 저장 → 별도 프로세스 재접속 → 검색을 확인했습니다.
질문 2건 모두 1순위 청크에 근거가 있었습니다: 한영대학교 사업의 130,000,000원·계약일로부터 3개월,
한국연구재단 UICC 사업의 SW 개발비 92,300,000원 이내. 답변 생성 LLM은 호출하지 않았습니다.

이는 소량 연결 확인입니다. 원본 전체 파싱·팀 최종 청킹·팀 전체 DB·검색 품질 평가·에이전트 통합은 남아 있습니다.
샘플 페이지 정보는 없어 페이지 인용 품질도 아직 검증하지 않았습니다.

```bash
# 자동 테스트는 테스트 전용 임베딩을 사용하며 OpenAI API를 호출하지 않습니다.
uv run pytest
```

자동 테스트는 입력 검사, 저장·재접속·필터, 재생성 충돌 방지, 모델 불일치, 실패한 인덱스 차단,
검색 결과의 출처 표시를 확인합니다. 테스트용 벡터 결과를 의미 검색 성능으로 해석하지 않습니다.
