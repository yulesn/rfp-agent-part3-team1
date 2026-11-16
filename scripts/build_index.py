"""[문서 처리 + RAG] 인덱스 생성 스크립트

원본 RFP → 파싱 → 청킹 → 임베딩 → Chroma 저장 까지 한 번에 실행합니다.
문서·파싱 방식·청킹 설정·임베딩 모델이 바뀌었을 때만 다시 실행하면 됩니다.

실행:  uv run python scripts/build_index.py --config configs/default.yaml
"""

import argparse
from pathlib import Path

from rfp_agent.config import load_config
from rfp_agent.ingest.chunking import chunk_documents
from rfp_agent.ingest.loader import (
    load_documents,
    save_documents,
    read_documents,
)
from rfp_agent.retrieval.vectorstore import build_vectorstore

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # 1. 파싱 결과 경로
    documents_path = (
        Path(config["paths"]["processed_dir"]) / "documents.jsonl"
    )

    # 2. 기존 파싱 결과가 있으면 재사용
    if documents_path.exists():
        docs = read_documents(documents_path)
        print(f"기존 파싱 결과 로드: {len(docs)}건")

    # 3. 파싱 결과가 없으면 원본 문서 파싱
    else:
        docs = load_documents(
            config["paths"]["raw_dir"],
            config["paths"]["metadata_file"]
        )
        save_documents(docs, documents_path)
        print(f"파싱 완료: {len(docs)}건")

    if not docs:
        raise ValueError("처리할 문서가 없습니다.")

    chunks = chunk_documents(docs, **config["chunking"])
    print(f"청킹 완료: {len(chunks)}개")

    build_vectorstore(chunks, config)
    print(f"벡터DB 저장 완료: {config['paths']['vectorstore_dir']}")


if __name__ == "__main__":
    main()
