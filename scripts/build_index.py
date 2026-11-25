
"""[문서 처리 + RAG] 인덱스 생성 스크립트

원본 RFP → 파싱 → 청킹 → 임베딩 → Chroma 저장 까지 한 번에 실행합니다.

- 기존 파싱 데이터가 있으면 재사용한다.
- 기존 파싱 데이터에 페이지 정보가 없으면 PDF만 재파싱한다.
- HWP/HWPX는 기존 파싱 데이터를 유지한다.
- 페이지 정보를 포함한 청크로 벡터 DB를 재구축한다.

실행: uv run python scripts/build_index.py --config configs/default.yaml
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
from rfp_agent.ingest.pdf import parse_pdf
from rfp_agent.ingest.schema import ParsedDocument
from rfp_agent.retrieval.vectorstore import build_vectorstore


def reparse_pdfs(
    docs: list[ParsedDocument],
    raw_dir: str | Path,
) -> list[ParsedDocument]:
    """기존 문서 중 PDF만 재파싱하여 페이지 정보를 추가한다."""

    raw_dir = Path(raw_dir)

    updated_docs = []
    pdf_count = 0

    for doc in docs:
        # 기존 HWP/HWPX 데이터는 그대로 재사용
        if Path(doc.doc_id).suffix.lower() != ".pdf":
            updated_docs.append(doc)
            continue

        pdf_path = raw_dir / doc.doc_id

        if not pdf_path.exists():
            print(f"PDF 원본 없음 - 기존 데이터 유지: {doc.doc_id}")
            updated_docs.append(doc)
            continue

        print(f"PDF 페이지 재파싱: {doc.doc_id}")

        text, pages = parse_pdf(pdf_path)

        updated_docs.append(
            ParsedDocument(
                doc_id=doc.doc_id,
                text=text,
                metadata=doc.metadata,
                pages=pages,
            )
        )

        pdf_count += 1

        print(f"  페이지 추출 완료: {len(pages)}페이지")

    print(f"PDF 재파싱 완료: {pdf_count}개")

    return updated_docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. 파싱 결과 경로
    documents_path = (
        Path(config["paths"]["processed_dir"]) / "documents.jsonl"
    )

    # 2. 기존 파싱 결과 재사용 또는 최초 전체 파싱
    if documents_path.exists():
        docs = read_documents(documents_path)
        print(f"기존 파싱 결과 로드: {len(docs)}건")

        # 3. 페이지 정보가 없는 기존 데이터라면 PDF만 재파싱
        pdf_docs = [
            doc for doc in docs
            if Path(doc.doc_id).suffix.lower() == ".pdf"
        ]

        missing_pages = any(not doc.pages for doc in pdf_docs)

        if missing_pages:
            print("기존 PDF 페이지 정보가 없어 PDF 재파싱을 진행합니다.")
            docs = reparse_pdfs(
                docs,
                config["paths"]["raw_dir"],
            )

            # 재파싱 결과 저장
            save_documents(docs, documents_path)
            print("페이지 정보가 포함된 파싱 결과 저장 완료")

        else:
            print("PDF 페이지 정보가 이미 존재합니다. 재파싱 생략")

    # 4. 파싱 결과가 없으면 원본 전체 파싱
    else:
        docs = load_documents(
            config["paths"]["raw_dir"],
            config["paths"]["metadata_file"],
        )
        save_documents(docs, documents_path)
        print(f"전체 문서 파싱 완료: {len(docs)}건")

    if not docs:
        raise ValueError("처리할 문서가 없습니다.")

    # 5. 전체 문서 청킹
    chunks = chunk_documents(docs, **config["chunking"])
    print(f"청킹 완료: {len(chunks)}개")

    # 6. 벡터 DB 재구축
    build_vectorstore(chunks, config)
    print(f"벡터DB 저장 완료: {config['paths']['vectorstore_dir']}")


if __name__ == "__main__":
    main()
