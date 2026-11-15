"""문서 처리 담당에게 받은 청크를 API 호출 전에 검사합니다."""

import hashlib
import json
import math
from datetime import date
from pathlib import Path


def validate_metadata(metadata: dict) -> str:
    """팀 전달 규칙을 검사하고 chunk_id를 반환합니다. 키 이름을 바꾸지 않습니다."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata는 dict여야 합니다.")
    for key in ("doc_id", "chunk_id"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"청크 metadata에 비어 있지 않은 {key} 문자열이 필요합니다.")
    prefix, separator, number = metadata["chunk_id"].rpartition("#")
    if (
        not separator
        or prefix != metadata["doc_id"]
        or not number.isascii()
        or not number.isdigit()
    ):
        raise ValueError("chunk_id는 doc_id#순번 형식이어야 합니다.")
    for key, value in metadata.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, (str, int, float, bool))
            or isinstance(value, float)
            and not math.isfinite(value)
        ):
            raise ValueError(f"지원하지 않는 metadata 값: {key}. 빈 값은 키를 생략하세요.")
    for key in ("사업 금액", "사업금액"):
        if key in metadata and (type(metadata[key]) is not int or metadata[key] < 0):
            raise ValueError(f"{key}은 원 단위 0 이상의 정수여야 합니다.")
    # 합의된 키 이름은 보존하며, 여기 나열된 날짜 키에 YYYY-MM-DD 검사를 적용합니다.
    for key in ("공개 일자", "입찰 참여 시작일", "입찰 참여 마감일", "공고일", "입찰마감일"):
        if key in metadata:
            value = metadata[key]
            try:
                if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"{key}은 YYYY-MM-DD 문자열이어야 합니다.") from exc
    return metadata["chunk_id"]


def read_chunks(path: Path) -> list[dict]:
    """page_content와 metadata가 있는 JSONL을 읽습니다. 잘못된 행을 묵인하지 않습니다."""
    records = []
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("각 행은 JSON 객체여야 합니다.")
                text = record.get("page_content")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("page_content는 비어 있지 않은 문자열이어야 합니다.")
                chunk_id = validate_metadata(record.get("metadata"))
                if chunk_id in seen:
                    raise ValueError("chunk_id가 중복되었습니다.")
                seen.add(chunk_id)
                records.append({"page_content": text, "metadata": record["metadata"]})
            except (ValueError, TypeError) as exc:
                # 원문이나 비밀 값이 오류 메시지에 섞이지 않도록 행 번호만 안내합니다.
                raise ValueError(f"청크 JSONL {number}행: {exc}") from exc
    if not records:
        raise ValueError("빈 청크 목록으로 인덱스를 만들 수 없습니다.")
    return records


def describe_chunks(path: Path, records: list[dict]) -> dict:
    return {
        "documents": len({row["metadata"]["doc_id"] for row in records}),
        "chunks": len(records),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata_keys": sorted({key for row in records for key in row["metadata"]}),
        "missing_pages": sum(not row["metadata"].get("page_num") for row in records),
        "api_called": False,
    }
