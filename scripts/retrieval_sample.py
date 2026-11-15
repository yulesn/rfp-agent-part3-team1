"""제공 CSV의 텍스트로 검색 연결을 먼저 확인합니다. PDF/HWP 파서 검증은 별도입니다."""

import argparse
import csv
import json
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rfp_agent.config import PROJECT_ROOT, load_config


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare(csv_path: Path, output: Path, config: dict, limit: int) -> dict:
    """샘플만 임시 청킹합니다. 팀 ingest.chunk_documents를 대체하지 않습니다."""
    if limit < 1:
        raise ValueError("limit는 1 이상이어야 합니다.")
    size = config["chunking"]["chunk_size"]
    overlap = config["chunking"]["chunk_overlap"]
    if type(size) is not int or type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("chunk_size > chunk_overlap >= 0인 정수를 사용하세요.")
    chunks = []
    seen = set()
    skipped = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"파일명", "텍스트"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV에 파일명과 텍스트 열이 필요합니다. UTF-8 CSV로 저장하세요.")
        for row in reader:
            doc_id = unicodedata.normalize("NFC", row.get("파일명") or "").strip()
            text = row.get("텍스트") or ""
            if not doc_id or not text.strip() or doc_id in seen:
                skipped += 1
                continue
            # 원본 한글 키/띄어쓰기를 유지하며 검색 모듈에서 키 이름을 바꾸지 않습니다.
            metadata = {
                key: value
                for key, value in row.items()
                if key and key != "텍스트" and value is not None and value.strip()
            }
            if "사업 금액" in metadata:
                amount = Decimal(metadata["사업 금액"].replace(",", ""))
                if not amount.is_finite() or amount != amount.to_integral_value():
                    raise ValueError("사업 금액은 원 단위 정수여야 합니다.")
                metadata["사업 금액"] = int(amount)
            metadata.update(doc_id=doc_id, source=doc_id)
            # 의존성 없이 준비할 수 있는 임시 고정 문자 수 분할입니다.
            # 실제 팀 파서/청킹 결과를 받을 때는 그 Document를 바로 build_vectorstore에 전달합니다.
            for number, start in enumerate(range(0, len(text), size - overlap)):
                content = text[start : start + size].strip()
                if content:
                    chunks.append(
                        {
                            "page_content": content,
                            "metadata": {**metadata, "chunk_id": f"{doc_id}#{number}"},
                        }
                    )
                if start + size >= len(text):
                    break
            seen.add(doc_id)
            if len(seen) == limit:
                break
    if len(seen) < limit:
        raise ValueError(f"유효한 문서가 {len(seen)}건입니다. 요청한 {limit}건보다 적습니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return {
        "documents": len(seen),
        "chunks": len(chunks),
        "skipped": skipped,
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrieval_sample.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="CSV 텍스트로 샘플 청크 JSONL 생성 (API 호출 없음)")
    prep.add_argument("--csv", required=True)
    prep.add_argument("--limit", type=int, default=3)
    commands.add_parser("build", help="샘플 청크를 임베딩하여 새 Chroma 컬렉션 생성")
    search = commands.add_parser("search", help="저장한 샘플 검색")
    search.add_argument("--query", required=True)
    search.add_argument("--agency", help="샘플 CSV의 발주 기관과 같은 값")
    search.add_argument("--min-budget", type=int, help="샘플 CSV의 사업 금액 최솟값(원)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    chunks_path = project_path(config["paths"]["processed_dir"]) / "chunks.jsonl"
    try:
        if args.command == "prepare":
            result = prepare(project_path(args.csv), chunks_path, config, args.limit)
        elif args.command == "build":
            from langchain_core.documents import Document

            from rfp_agent.retrieval.vectorstore import build_vectorstore

            with chunks_path.open(encoding="utf-8") as stream:
                chunks = [Document(**json.loads(line)) for line in stream if line.strip()]
            build_vectorstore(chunks, config)
            result = {"chunks": len(chunks), "status": "index_ready"}
        else:
            from rfp_agent.retrieval.retriever import retrieve

            clauses = []
            if args.agency:
                clauses.append({"발주 기관": args.agency})
            if args.min_budget is not None:
                clauses.append({"사업 금액": {"$gte": args.min_budget}})
            filters = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)
            docs = retrieve(args.query, config, filters)
            result = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    except (ValueError, OSError, InvalidOperation) as exc:
        parser.exit(1, f"실행 실패: {exc}\n")
    except Exception as exc:  # noqa: BLE001 -- CLI 경계에서 외부 오류 본문을 숨깁니다.
        # 외부 API 오류 응답은 키를 노출하지 않도록 종류와 상태만 표시합니다.
        parser.exit(1, f"실행 실패: {type(exc).__name__}, HTTP {getattr(exc, 'status_code', None)}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
