"""파싱 결과 JSONL을 청킹해 저장합니다. 임베딩 API는 호출하지 않습니다."""

import argparse
import hashlib
import json
from pathlib import Path

from rfp_agent.config import load_config
from rfp_agent.ingest.chunking import chunk_documents
from rfp_agent.ingest.schema import ParsedDocument
from rfp_agent.retrieval.index_io import describe_chunks, normalize_metadata


def read_parsed_documents(path: Path) -> list[ParsedDocument]:
    """전체 본문과 문서 ID를 확인하며 읽습니다. 파서는 다시 실행하지 않습니다."""
    documents = []
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("각 행은 JSON 객체여야 합니다.")
                doc_id, text = row.get("doc_id"), row.get("text")
                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise ValueError("doc_id가 필요합니다.")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("본문은 비어 있지 않은 문자열이어야 합니다.")
                if doc_id in seen:
                    raise ValueError("doc_id가 중복되었습니다.")
                metadata = row.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise TypeError("metadata는 dict여야 합니다.")
                documents.append(ParsedDocument(doc_id=doc_id, text=text, metadata=metadata))
                seen.add(doc_id)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"파싱 JSONL {line_number}행: {exc}") from exc
    if not documents:
        raise ValueError("문서가 없습니다.")
    return documents


def export_chunks(documents_path: Path, output_path: Path, config: dict) -> dict:
    """모든 청크를 검사한 후 새 파일에 저장합니다. 기존 결과는 덮어쓰지 않습니다."""
    if output_path.exists():
        raise FileExistsError("청크 파일이 이미 있습니다. 새 출력 경로를 사용하세요.")
    documents = read_parsed_documents(documents_path)
    chunks = chunk_documents(documents, **config["chunking"])
    records = []
    seen = set()
    for chunk in chunks:
        metadata = normalize_metadata(chunk.metadata)
        if metadata["chunk_id"] in seen:
            raise ValueError("chunk_id가 중복되었습니다.")
        seen.add(metadata["chunk_id"])
        records.append({"page_content": chunk.page_content, "metadata": metadata})
    if not records:
        raise ValueError("검색에 사용할 청크가 없습니다.")
    if {record["metadata"]["doc_id"] for record in records} != {doc.doc_id for doc in documents}:
        raise ValueError("청킹 후 검색할 본문이 남지 않은 문서가 있습니다.")
    # UTF-8 변환까지 확인한 뒤 파일을 만듭니다. 잘못된 문자열로 부분 저장하지 않습니다.
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(content)
    return {
        **describe_chunks(output_path, records),
        "parsed_documents": len(documents),
        "documents_sha256": hashlib.sha256(documents_path.read_bytes()).hexdigest(),
        "chunking": config["chunking"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = export_chunks(Path(args.documents), Path(args.out), load_config(args.config))
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
