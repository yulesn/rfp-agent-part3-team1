"""[검색] 벡터DB (Chroma)

- 별도 서버 없이 Chroma 를 로컬 폴더(data/vectorstore)에 파일로 저장해서 씁니다.
- build_vectorstore: 청크 목록을 임베딩해서 저장 (scripts/build_index.py 에서 호출)
- load_vectorstore: 저장된 벡터DB 를 불러오기 (에이전트·평가에서 호출)
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rfp_agent.retrieval.embeddings import get_embeddings


def build_vectorstore(chunks: list[Document], config: dict) -> Chroma:
    # TODO: Chroma.from_documents(chunks, embedding=get_embeddings(config),
    #       collection_name=..., persist_directory=config["paths"]["vectorstore_dir"])
    raise NotImplementedError


def load_vectorstore(config: dict) -> Chroma:
    return Chroma(
        collection_name=config["vectorstore"]["collection_name"],
        embedding_function=get_embeddings(config),
        persist_directory=config["paths"]["vectorstore_dir"],
    )
