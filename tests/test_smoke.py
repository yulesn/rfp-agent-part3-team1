"""기본 동작 확인 테스트. 실행: uv run pytest

각 파트에서 함수를 완성하면 간단한 테스트를 추가해 주세요 (예: 샘플 PDF 1개 파싱 결과가 비어 있지 않은지).
"""

from rfp_agent.config import load_config
from rfp_agent.evaluation.metrics import hit_at_k


def test_load_config():
    config = load_config()
    assert "llm" in config and "retrieval" in config


def test_hit_at_k():
    assert hit_at_k(["a.hwp", "b.pdf"], "b.pdf") == 1.0
    assert hit_at_k(["a.hwp"], "b.pdf") == 0.0


def test_imports():
    # 모든 모듈이 문법 오류 없이 import 되는지 확인
    import rfp_agent.agent.graph
    import rfp_agent.ingest.loader
    import rfp_agent.retrieval.retriever  # noqa: F401


def test_api_health():
    from fastapi.testclient import TestClient

    from rfp_agent.api.main import app

    assert TestClient(app).get("/health").json() == {"status": "ok"}
