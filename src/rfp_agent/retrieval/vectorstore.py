"""[RAG] 청크를 Chroma에 저장하고, 같은 임베딩 설정으로 다시 불러옵니다.

이 베이스라인은 새 컬렉션을 한 번 생성합니다. 재구축할 때는 경로나 컬렉션명을
바꿔 이전 인덱스를 보관하세요. 미완성 인덱스와 모델이 다른 인덱스는 검색하지 않습니다.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rfp_agent.config import PROJECT_ROOT
from rfp_agent.retrieval.embeddings import get_embeddings
from rfp_agent.retrieval.index_io import validate_metadata


def _location(config: dict) -> tuple[Path, str]:
    path = Path(config["paths"]["vectorstore_dir"]).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(), config["vectorstore"]["collection_name"]


def _client(path: Path):
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def _embedding_metadata(config: dict) -> dict:
    return {
        "rfp_embedding_provider": config["embedding"]["provider"],
        "rfp_embedding_model": config["embedding"]["model"],
    }


def _validate_chunks(chunks: list[Document]) -> list[str]:
    if not chunks:
        raise ValueError("빈 청크 목록으로 인덱스를 만들 수 없습니다.")
    ids = []
    for chunk in chunks:
        if not chunk.page_content.strip():
            raise ValueError("청크 본문은 비어 있을 수 없습니다.")
        ids.append(validate_metadata(chunk.metadata))
    if len(set(ids)) != len(ids):
        raise ValueError("chunk_id가 중복되었습니다.")
    return ids


def build_vectorstore(chunks: list[Document], config: dict) -> Chroma:
    """청크 전체를 새 컬렉션에 저장합니다. 기존 컬렉션을 덮어쓰지 않습니다."""
    ids = _validate_chunks(chunks)
    path, name = _location(config)
    if (path / "chroma.sqlite3").is_file():
        client = _client(path)
        try:
            client.get_collection(name, embedding_function=None)
        except NotFoundError:
            pass
        else:
            raise FileExistsError("이미 존재하는 컬렉션입니다. 새 저장 경로나 컬렉션명을 쓰세요.")

    embeddings = get_embeddings(config)
    client = _client(path)
    metadata = {**_embedding_metadata(config), "rfp_index_state": "building"}
    collection = client.create_collection(name, metadata=metadata, embedding_function=None)
    store = Chroma(
        client=client,
        collection_name=name,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )
    # 전체 문서에서 청크 수가 Chroma의 1회 저장 한도를 넘을 수 있습니다.
    batch_size = client.get_max_batch_size()
    for start in range(0, len(chunks), batch_size):
        store.add_documents(chunks[start : start + batch_size], ids=ids[start : start + batch_size])
    collection.modify(metadata={**metadata, "rfp_index_state": "ready"})
    return store


def load_vectorstore(config: dict) -> Chroma:
    """완성된 인덱스만 엽니다. DB 부재를 빈 검색 결과로 처리하지 않습니다."""
    path, name = _location(config)
    if not (path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"벡터DB가 없습니다. 인덱스를 먼저 생성하세요: {path}")
    client = _client(path)
    try:
        collection = client.get_collection(name, embedding_function=None)
    except NotFoundError as exc:
        raise FileNotFoundError(f"컬렉션이 없습니다: {name}") from exc
    metadata = collection.metadata or {}
    if metadata.get("rfp_index_state") != "ready":
        raise ValueError("완료 표시가 없는 인덱스입니다. 새 경로에 인덱스를 생성하세요.")
    if any(metadata.get(key) != value for key, value in _embedding_metadata(config).items()):
        raise ValueError("인덱스와 임베딩 모델 설정이 다릅니다. 같은 설정으로 검색하세요.")
    return Chroma(
        client=client,
        collection_name=name,
        embedding_function=get_embeddings(config),
        create_collection_if_not_exists=False,
    )
