"""[프론트·백엔드·평가] FastAPI 서버

- 에이전트(build_graph)를 HTTP API로 감싸는 파일입니다. UI와 평가 스크립트는 이 API를 호출합니다.
- 그래프와 벡터DB는 서버가 켜질 때 한 번만 만들고, 요청마다 재사용합니다.
- API 서버가 준비되기 전까지는 Streamlit 이 build_graph() 를 직접 불러도 됩니다.

실행:  uv run uvicorn rfp_agent.api.main:app --reload --port 8000
확인:  브라우저에서 http://localhost:8000/docs  (자동 생성 API 문서에서 바로 테스트 가능)
"""

from fastapi import FastAPI

from rfp_agent.api.schemas import AskRequest, AskResponse

app = FastAPI(title="RFP RAG 에이전트 API")


@app.get("/health")
def health() -> dict:
    """서버가 살아 있는지 확인"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # TODO:
    # 1. 서버 시작 시 graph = build_graph(load_config()) 를 한 번만 만들어 두기
    # 2. graph.invoke({"messages": [("user", req.question)]}) 실행
    # 3. 마지막 메시지 → answer, 도구 결과에서 출처 정보 → sources 로 담아 반환
    raise NotImplementedError
