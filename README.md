# RFP RAG 에이전트

공공 입찰 RFP(제안요청서, hwp·pdf) 문서를 검색해서 질문에 답하고, 조건에 맞는 공고를 찾고, 공고의 핵심 내용을 요약해 주는 LangGraph 기반 RAG 에이전트입니다.

- 팀 노션: [RFP RAG 에이전트](https://app.notion.com/p/413232ca6c7b812099f3f7e9436693d2) (작업 방식, 기록 규칙, 문제 해결은 노션 「🛠️ 개발 가이드」 참고)

## 초기 설정

```bash
# 1. uv 설치 (macOS/Linux, Windows는 https://docs.astral.sh/uv/ 참고)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 레포 받기 + 패키지 설치
git clone https://github.com/yulesn/rfp-agent-part3-team1.git
cd rfp-agent-part3-team1
uv sync

# 3. API 키 설정: .env 를 열어 OPENAI_API_KEY 입력
cp .env.example .env

# 4. 데이터: RFP 원본과 메타데이터 파일을 data/raw/ 에 넣기 (git에 올리지 않음)

# 5. 확인
uv run pytest
```

VS Code·Jupyter 에서는 인터프리터/커널로 `.venv` 를 선택하세요.

## uv 규칙

| 언제 | 명령 |
| --- | --- |
| `git pull` 한 뒤 | `uv sync` |
| 패키지 추가 | `uv add 패키지명` (개발용은 `uv add --dev`) → `pyproject.toml`, `uv.lock` 함께 커밋 |
| 패키지 삭제 | `uv remove 패키지명` |
| 코드 실행 | `uv run python ...`, `uv run pytest` |

`pip install` 은 쓰지 않습니다.

## 실행

```bash
uv run python scripts/build_index.py      # 문서 → 벡터DB 생성 (문서·파싱·청킹·임베딩 설정이 바뀔 때만)
uv run python scripts/ask.py "질문"        # 터미널에서 질문
uv run streamlit run app/streamlit_app.py # 데모 UI
uv run uvicorn rfp_agent.api.main:app --reload --port 8000  # API 서버 (문서: localhost:8000/docs)
uv run python scripts/evaluate.py         # 평가
```

다른 설정으로 실험할 때는 `configs/default.yaml` 을 복사해서 `--config configs/새설정.yaml` 로 실행합니다.

## 프로젝트 구조

```
├── configs/default.yaml      # 실험 설정 (청크 크기, 모델명, top_k 등)
├── data/                     # 데이터 (git 제외): raw/, processed/, vectorstore/
├── src/rfp_agent/
│   ├── config.py, llm.py     # 설정 불러오기, LLM 생성
│   ├── ingest/               # 문서 파싱·메타데이터·청킹
│   ├── retrieval/            # 임베딩·벡터DB(Chroma)·검색
│   ├── agent/                # LangGraph 에이전트 (state, prompts, nodes, graph)
│   │   └── tools/            #   에이전트 도구 (도구 1개 = 파일 1개)
│   ├── api/                  # FastAPI 서버 (에이전트를 API로 제공)
│   └── evaluation/           # 평가 지표
├── scripts/                  # build_index, ask, evaluate
├── eval/questions.jsonl      # 평가용 질문·정답
├── app/streamlit_app.py      # 데모 UI
├── notebooks/                # 개인 실험 노트북 (이름: 01_이름_내용.ipynb)
└── tests/
```

벡터DB는 별도 서버 없이 Chroma 를 `data/vectorstore/` 에 파일로 저장합니다.

## 파트별 담당 파일

각 파일 맨 위 주석에 작성할 내용이 적혀 있습니다. `TODO` 부분을 채워 주세요.

| 파트 (담당) | 파일 |
| --- | --- |
| 문서 처리 (원재민) | `ingest/schema.py`, `pdf.py`, `hwp.py`, `metadata.py`, `loader.py`, `chunking.py`, `scripts/build_index.py` |
| RAG (김원태) | `retrieval/embeddings.py`, `vectorstore.py`, `retriever.py`, `agent/tools/search_rfp.py` |
| LLM·에이전트 (유영관) | `agent/state.py`, `prompts.py`, `nodes.py`, `graph.py`, `agent/tools/filter_metadata.py`, `summarize_rfp.py`, `scripts/ask.py` |
| 클라우드 인프라·오픈소스 모델 (신유정) | `llm.py` (OpenAI / 오픈소스 모델 전환), GCP VM·L4 GPU 모델 서빙 |
| 프론트·백엔드·평가 (조영권) | `app/streamlit_app.py`, `api/main.py`, `api/schemas.py`, `eval/questions.jsonl`, `evaluation/metrics.py`, `scripts/evaluate.py` |
| 공통 | `config.py`, `configs/default.yaml`, `tests/` |

## 파트 간 약속

바꿔야 하면 팀에 먼저 공유해 주세요.

| 경계 | 형식 |
| --- | --- |
| 파싱 → 검색 | `ParsedDocument(doc_id, text, metadata)` (`ingest/schema.py`) |
| 청킹 → 벡터DB | `Document(page_content, metadata)`, metadata 에 원본 메타데이터 포함 |
| 검색 → 에이전트 | `retrieve(query, config, filters=None) -> list[Document]` (`retrieval/retriever.py`) |
| 에이전트 → API·UI·평가 | `build_graph(config).invoke({"messages": [("user", 질문)]})` (`agent/graph.py`) |
| API → UI·평가 | `POST /ask` 요청 `AskRequest`, 응답 `AskResponse(answer, sources)` (`api/schemas.py`) |

- 숫자·모델명은 코드에 쓰지 말고 `configs/*.yaml` 에서 불러오기
- LLM 은 `get_llm(config)`, 임베딩은 `get_embeddings(config)` 로만 생성
- `.env`, `data/` 는 커밋하지 않기
