"""제공 CSV가 샘플 검색 입력으로 안전하게 전달되는지 확인합니다."""

import csv
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retrieval_sample.py"
spec = importlib.util.spec_from_file_location("retrieval_sample", SCRIPT)
sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["파일명", "텍스트", "발주 기관", "사업 금액", "빈 값"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_preserves_source_and_numeric_filter_metadata(tmp_path):
    source = tmp_path / "source.csv"
    doc_id = "한영대학교.hwp"
    write_csv(
        source,
        [
            {
                "파일명": unicodedata.normalize("NFD", doc_id),
                "텍스트": "학사 요구사항" * 4,
                "발주 기관": "한영대학교",
                "사업 금액": "130,000,000",
            }
        ],
    )
    output = tmp_path / "chunks.jsonl"
    result = sample.prepare(source, output, {"chunking": {"chunk_size": 12, "chunk_overlap": 3}}, 1)
    chunks = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["documents"] == 1 and result["chunks"] > 1
    assert all(chunk["metadata"]["doc_id"] == doc_id for chunk in chunks)
    assert all(chunk["metadata"]["사업 금액"] == 130000000 for chunk in chunks)
    assert "빈 값" not in chunks[0]["metadata"]
    assert len({chunk["metadata"]["chunk_id"] for chunk in chunks}) == len(chunks)


def test_prepare_refuses_insufficient_real_documents(tmp_path):
    source = tmp_path / "source.csv"
    write_csv(source, [{"파일명": "a.hwp", "텍스트": ""}])
    output = tmp_path / "chunks.jsonl"
    with pytest.raises(ValueError, match="유효한 문서"):
        sample.prepare(source, output, {"chunking": {"chunk_size": 1000, "chunk_overlap": 200}}, 3)
    assert not output.exists()


def test_prepare_does_not_silently_round_budget_or_overwrite(tmp_path):
    source = tmp_path / "source.csv"
    write_csv(source, [{"파일명": "a.hwp", "텍스트": "본문", "사업 금액": "1.5"}])
    config = {"chunking": {"chunk_size": 1000, "chunk_overlap": 200}}
    output = tmp_path / "chunks.jsonl"
    with pytest.raises(ValueError, match="정수"):
        sample.prepare(source, output, config, 1)
    write_csv(source, [{"파일명": "a.hwp", "텍스트": "본문", "사업 금액": "1"}])
    sample.prepare(source, output, config, 1)
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        sample.prepare(source, output, config, 1)
    assert output.read_bytes() == saved
