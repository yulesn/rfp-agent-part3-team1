"""[검색] 임베딩 모델

- config["embedding"] 에 따라 임베딩 객체를 만들어 돌려줍니다.
- 기본은 OpenAI text-embedding-3-small. 한국어 오픈소스 임베딩(예: bge-m3, ko-sroberta)과 비교 실험해 보고
  결과는 노션 「실험 로그」(분류: 임베딩)에 남겨 주세요.
- 주의: 임베딩 모델을 바꾸면 벡터DB를 다시 만들어야 합니다 (scripts/build_index.py 재실행).
"""

from langchain_core.embeddings import Embeddings


def get_embeddings(config: dict) -> Embeddings:
    emb_cfg = config["embedding"]
    if emb_cfg["provider"] == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=emb_cfg["model"])

    # TODO: huggingface 임베딩 추가 시 여기에 (langchain-huggingface 패키지는 `uv add` 로 추가)
    raise ValueError(f"지원하지 않는 embedding provider: {emb_cfg['provider']}")
