"""[문서 파싱 + 검색] 인덱스 생성 스크립트

원본 RFP → 파싱 → 청킹 → 임베딩 → Chroma 저장 까지 한 번에 실행합니다.
문서·파싱 방식·청킹 설정·임베딩 모델이 바뀌었을 때만 다시 실행하면 됩니다.

실행:  uv run python scripts/build_index.py --config configs/default.yaml
"""

import argparse

from rfp_agent.config import load_config
from rfp_agent.ingest.chunking import chunk_documents
from rfp_agent.ingest.loader import load_documents, save_documents
from rfp_agent.retrieval.vectorstore import build_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    docs = load_documents(config["paths"]["raw_dir"], config["paths"]["metadata_file"])
    save_documents(docs, f"{config['paths']['processed_dir']}/documents.jsonl")
    print(f"파싱 완료: {len(docs)}건")

    chunks = chunk_documents(docs, **config["chunking"])
    print(f"청킹 완료: {len(chunks)}개")

    build_vectorstore(chunks, config)
    print(f"벡터DB 저장 완료: {config['paths']['vectorstore_dir']}")


if __name__ == "__main__":
    main()
