"""청킹 결과를 저장하기 전에 잘못된 입력과 기존 파일을 보호합니다."""

import json

import pytest

from scripts.export_chunks import export_chunks


def save_documents(tmp_path, rows):
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path


def test_duplicate_document_ids_do_not_create_partial_output(tmp_path):
    row = {"doc_id": "사업.hwp", "text": "사업 본문", "metadata": {}}
    source = save_documents(tmp_path, [row, row])
    output = tmp_path / "chunks.jsonl"
    with pytest.raises(ValueError, match="2행.*중복"):
        export_chunks(source, output, {"chunking": {"chunk_size": 1000, "chunk_overlap": 200}})
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "chunks.jsonl"
    output.write_bytes(b"existing result")
    with pytest.raises(FileExistsError):
        export_chunks(tmp_path / "not-needed.jsonl", output, {})
    assert output.read_bytes() == b"existing result"


def test_later_bad_metadata_does_not_leave_partial_chunks(tmp_path):
    source = save_documents(tmp_path, [
        {"doc_id": "a.hwp", "text": "첫 문서 본문", "metadata": {}},
        {"doc_id": "b.hwp", "text": "둘째 문서 본문", "metadata": {"사업 금액": "1억원"}},
    ])
    before = source.read_bytes()
    output = tmp_path / "chunks.jsonl"
    with pytest.raises(ValueError, match="원 단위"):
        export_chunks(source, output, {"chunking": {"chunk_size": 1000, "chunk_overlap": 200}})
    assert not output.exists()
    assert source.read_bytes() == before
