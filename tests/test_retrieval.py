"""실제 Chroma 저장·검색 계약을 검증합니다. 테스트 벡터는 의미 검색 성능을 평가하지 않습니다."""

from copy import deepcopy
from importlib import import_module

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rfp_agent.retrieval import vectorstore
from rfp_agent.retrieval.embeddings import get_embeddings
from rfp_agent.retrieval.retriever import retrieve


class KeywordEmbeddings(Embeddings):
    """API를 호출하지 않는 테스트 전용 벡터. 제품 설정에서는 사용할 수 없습니다."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count(word)) for word in ("학사", "산학", "예산")] + [1.0]


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda config: KeywordEmbeddings())
    return {
        "paths": {"vectorstore_dir": str(tmp_path / "index")},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
        "vectorstore": {"collection_name": "test_rfp"},
        "retrieval": {"top_k": 2},
    }


@pytest.fixture
def chunks():
    return [
        Document(
            page_content="학사 시스템 예산 1억 원",
            metadata={
                "doc_id": "a.hwp",
                "chunk_id": "a.hwp#0",
                "발주 기관": "A대학",
                "사업 금액": 100000000,
            },
        ),
        Document(
            page_content="산학 시스템 예산 6억 원",
            metadata={
                "doc_id": "b.pdf",
                "chunk_id": "b.pdf#0",
                "발주 기관": "B재단",
                "사업 금액": 600000000,
            },
        ),
        Document(
            page_content="학사 고도화 일정",
            metadata={
                "doc_id": "a.hwp",
                "chunk_id": "a.hwp#1",
                "발주 기관": "A대학",
                "사업 금액": 100000000,
            },
        ),
    ]


def test_persistent_search_and_filters(config, chunks, monkeypatch, tmp_path):
    vectorstore.build_vectorstore(chunks, config)
    monkeypatch.chdir(tmp_path)  # 작업 디렉터리가 바뀌어도 절대 경로로 재접속
    docs = retrieve("학사", config)
    assert len(docs) == 2
    assert all(isinstance(doc, Document) and doc.metadata["doc_id"] == "a.hwp" for doc in docs)
    assert retrieve("예산", config, {"발주 기관": "B재단"})[0].metadata["doc_id"] == "b.pdf"
    filters = {"$and": [{"사업 금액": {"$gte": 500000000}}, {"발주 기관": "B재단"}]}
    assert len(retrieve("예산", config, filters)) == 1
    assert retrieve("예산", config, {"발주 기관": "없는 기관"}) == []
    assert len(retrieve("학사", config, {})) == 2


def test_upstream_metadata_survives_storage_and_retrieval(config, chunks):
    for chunk in chunks:
        old_id = chunk.metadata.pop("chunk_id")
        index = int(old_id.rsplit("#", 1)[1])
        chunk.metadata["chunk_index"] = index * 4
        chunk.metadata["입찰 참여 마감일"] = "2026-11-30 18:01:59"
    before = deepcopy(chunks)
    store = vectorstore.build_vectorstore(chunks, config)
    assert set(store.get()["ids"]) == {"a.hwp#0", "a.hwp#4", "b.pdf#0"}
    documents = retrieve("학사", config)
    assert {doc.metadata["chunk_id"] for doc in documents} == {"a.hwp#0", "a.hwp#4"}
    for document in documents:
        assert document.metadata["입찰 참여 마감일"] == "2026-11-30 18:01:59"
        assert "발주 기관" in document.metadata
        assert "발주기관" not in document.metadata
        assert document.metadata["chunk_index"] in (0, 4)
    assert chunks == before


def test_rebuild_does_not_overwrite_or_duplicate(config, chunks):
    original = vectorstore.build_vectorstore(chunks, config)
    with pytest.raises(FileExistsError):
        vectorstore.build_vectorstore(chunks, config)
    assert len(original.get()["ids"]) == 3


def test_embedding_model_mismatch(config, chunks):
    vectorstore.build_vectorstore(chunks, config)
    changed = deepcopy(config)
    changed["embedding"]["model"] = "other-model"
    with pytest.raises(ValueError, match="모델 설정"):
        retrieve("학사", changed)


def test_failed_build_cannot_be_searched(config, chunks, monkeypatch):
    class BrokenEmbeddings(KeywordEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("simulated API failure")

    monkeypatch.setattr(vectorstore, "get_embeddings", lambda config: BrokenEmbeddings())
    with pytest.raises(RuntimeError):
        vectorstore.build_vectorstore(chunks, config)
    with pytest.raises(ValueError, match="완료 표시"):
        retrieve("학사", config)


def test_missing_db_does_not_create_empty_index(config):
    with pytest.raises(FileNotFoundError):
        retrieve("학사", config)


@pytest.mark.parametrize(
    "problem",
    ["empty", "duplicate", "missing_id", "nested", "nan", "index_conflict", "index_duplicate", "invalid_time"],
)
def test_invalid_chunks_fail_before_embedding(config, chunks, problem, monkeypatch):
    if problem == "empty":
        chunks = []
    elif problem == "duplicate":
        chunks.append(chunks[0])
    elif problem == "missing_id":
        del chunks[0].metadata["doc_id"]
    elif problem == "nested":
        chunks[0].metadata["bad"] = ["list"]
    elif problem == "nan":
        chunks[0].metadata["bad"] = float("nan")
    elif problem == "index_conflict":
        chunks[0].metadata["chunk_index"] = 1
    elif problem == "index_duplicate":
        chunks[1].metadata = {"doc_id": "a.hwp", "chunk_index": 0}
    else:
        chunks[0].metadata["입찰 참여 마감일"] = "2026-11-30 24:00:00"
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda _: pytest.fail("API 전 검증 필요"))
    with pytest.raises(ValueError):
        vectorstore.build_vectorstore(chunks, config)


@pytest.mark.parametrize(
    "query, top_k, filters",
    [
        (" ", 5, None),
        ("질문", 0, None),
        ("질문", True, None),
        ("질문", 5, {"사업 금액": {"$gte": "5억"}}),
        ("질문", 5, {"발주 기관": "A", "사업 금액": 1}),
    ],
)
def test_invalid_requests_fail_before_db(config, query, top_k, filters):
    config["retrieval"]["top_k"] = top_k
    with pytest.raises(ValueError):
        retrieve(query, config, filters)


def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_embeddings({"embedding": {"provider": "openai", "model": "text-embedding-3-small"}})


def test_openai_requests_respect_batch_size(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    embeddings = get_embeddings({"embedding": {
        "provider": "openai", "model": "text-embedding-3-small", "batch_size": 2,
    }})
    calls = []

    class SmallBatchClient:
        def create(self, *, input, **kwargs):
            assert len(input) <= 2
            calls.append(list(input))
            return {"data": [{"embedding": [1.0, 0.0]} for _ in input]}

    embeddings.client = SmallBatchClient()
    # API나 토크나이저 다운로드 없이 요청 분할만 확인합니다.
    embeddings.check_embedding_ctx_length = False
    assert len(embeddings.embed_documents(["학사 시스템"] * 5)) == 5
    assert [len(batch) for batch in calls] == [2, 2, 1]


@pytest.mark.parametrize("size", [0, -1, True, "128"])
def test_invalid_embedding_batch_size(size):
    with pytest.raises(ValueError, match="batch_size"):
        get_embeddings({"embedding": {
            "provider": "openai", "model": "text-embedding-3-small", "batch_size": size,
        }})


def test_search_tool_formats_sources_and_empty_result(monkeypatch):
    module = import_module("rfp_agent.agent.tools.search_rfp")
    monkeypatch.setattr(module, "load_config", dict)
    monkeypatch.setattr(
        module,
        "retrieve",
        lambda *args: [
            Document(
                page_content="원문 근거",
                metadata={"doc_id": "a.pdf", "chunk_id": "a.pdf#1", "page_num": 2},
            )
        ],
    )
    result = module.search_rfp.invoke({"query": "예산"})
    assert "원문 근거" in result and "a.pdf#1" in result and "페이지: 2" in result
    monkeypatch.setattr(module, "retrieve", lambda *args: [])
    assert "검색된 근거가 없습니다" in module.search_rfp.invoke({"query": "예산"})
