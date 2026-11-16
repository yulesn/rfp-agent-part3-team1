"""공유 화면의 입력·결과 표시를 검사합니다. API를 호출하지 않습니다."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from rfp_agent.retrieval import retriever

APP = Path(__file__).resolve().parents[1] / "app" / "retrieval_demo.py"


@pytest.fixture
def screen(tmp_path, monkeypatch):
    database = tmp_path / "db"
    database.mkdir()
    config = tmp_path / "demo.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"vectorstore_dir": str(database)},
                "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
                "retrieval": {"top_k": 5},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", [str(APP), "--config", str(config)])
    monkeypatch.setattr("rfp_agent.config.load_dotenv", lambda *_: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    search = Mock(side_effect=AssertionError("화면을 여는 것만으로 검색하면 안 됩니다."))
    monkeypatch.setattr(retriever, "retrieve", search)
    return database, search


def test_screen_opens_without_api_key_or_db(screen):
    _, search = screen
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.button[0].disabled
    assert len(app.warning) == 2
    search.assert_not_called()


def test_submit_searches_and_displays_evidence(screen, monkeypatch):
    database, search = screen
    (database / "chroma.sqlite3").touch()
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    search.side_effect = None
    search.return_value = [
        Document(
            page_content="사업 예산은 150,000,000원이다.",
            metadata={"doc_id": "demo_library.txt", "chunk_id": "demo_library.txt#0"},
        )
    ]
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert any("질문을 입력" in item.value for item in app.warning)
    search.assert_not_called()
    app.text_input[0].set_value("도서관 사업 예산은?")
    app.button[0].click().run()
    assert not app.exception
    search.assert_called_once()
    assert search.call_args.args[0] == "도서관 사업 예산은?"
    text = "\n".join(item.value for item in [*app.markdown, *app.caption])
    assert "150,000,000원" in text and "demo_library.txt#0" in text


def test_screen_does_not_display_api_error_body(screen, monkeypatch):
    database, search = screen
    (database / "chroma.sqlite3").touch()
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    search.side_effect = RuntimeError("sensitive-response-body")
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("예산은?")
    app.button[0].click().run()
    assert not app.exception
    text = "\n".join(item.value for item in [*app.markdown, *app.caption, *app.error])
    assert "RuntimeError" in text and "sensitive-response-body" not in text
