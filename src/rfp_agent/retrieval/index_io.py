"""문서 처리 담당에게 받은 청크를 API 호출 전에 검사합니다."""

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path


def normalize_metadata(metadata: dict) -> dict:
    """수령한 메타데이터를 검사하고 검색용 ID를 보완한 복사본을 반환합니다."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata는 dict여야 합니다.")
    if not isinstance(metadata.get("doc_id"), str) or not metadata["doc_id"].strip():
        raise ValueError("청크 metadata에 비어 있지 않은 doc_id 문자열이 필요합니다.")
    result = dict(metadata)
    if "chunk_index" in result:
        if type(result["chunk_index"]) is not int or result["chunk_index"] < 0:
            raise ValueError("chunk_index는 0 이상의 정수여야 합니다.")
        expected_id = f"{result['doc_id']}#{result['chunk_index']}"
        if "chunk_id" not in result:
            # 표 구분선 제거로 순번에 빈 구간이 생겨도 원래 순번을 유지합니다.
            result["chunk_id"] = expected_id
        elif result["chunk_id"] != expected_id:
            raise ValueError("chunk_id와 doc_id/chunk_index가 일치하지 않습니다.")
    if not isinstance(result.get("chunk_id"), str) or not result["chunk_id"].strip():
        raise ValueError("청크 metadata에 chunk_id 또는 chunk_index가 필요합니다.")
    prefix, separator, number = result["chunk_id"].rpartition("#")
    if (
        not separator
        or prefix != result["doc_id"]
        or not number.isascii()
        or not number.isdigit()
    ):
        raise ValueError("chunk_id는 doc_id#순번 형식이어야 합니다.")
    for key, value in result.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, (str, int, float, bool))
            or isinstance(value, float)
            and not math.isfinite(value)
        ):
            raise ValueError(f"지원하지 않는 metadata 값: {key}. 빈 값은 키를 생략하세요.")
    for key in ("사업 금액", "사업금액"):
        if key in result and (type(result[key]) is not int or result[key] < 0):
            raise ValueError(f"{key}은 원 단위 0 이상의 정수여야 합니다.")
    # 파서가 보존한 입찰 시각을 자르거나 다른 문자열로 바꾸지 않습니다.
    for key in ("공개 일자", "입찰 참여 시작일", "입찰 참여 마감일", "공고일", "입찰마감일"):
        if key in result:
            value = result[key]
            try:
                if isinstance(value, str) and len(value) == 10:
                    valid = date.fromisoformat(value).isoformat() == value
                elif isinstance(value, str) and len(value) == 19:
                    valid = datetime.fromisoformat(value).isoformat(sep=" ") == value
                else:
                    valid = False
                if not valid:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    f"{key}은 YYYY-MM-DD 또는 YYYY-MM-DD HH:MM:SS 문자열이어야 합니다."
                ) from exc
    return result


def validate_metadata(metadata: dict) -> str:
    """기존 호출부를 위해 입력을 변경하지 않고 검색용 ID를 반환합니다."""
    return normalize_metadata(metadata)["chunk_id"]


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
                metadata = normalize_metadata(record.get("metadata"))
                chunk_id = metadata["chunk_id"]
                if chunk_id in seen:
                    raise ValueError("chunk_id가 중복되었습니다.")
                seen.add(chunk_id)
                records.append({"page_content": text, "metadata": metadata})
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
