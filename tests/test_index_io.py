"""팀 청크를 받은 뒤 API 비용이 발생하기 전에 입력 계약을 검증합니다."""

import json

import pytest

from rfp_agent.retrieval.index_io import describe_chunks, read_chunks


def row(**metadata):
    return {
        "page_content": "사업 예산은 1억원입니다.",
        "metadata": {"doc_id": "사업.hwp", "chunk_id": "사업.hwp#0", **metadata},
    }


def save(tmp_path, rows):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rows), encoding="utf-8"
    )
    return path


def test_team_metadata_remains_unchanged(tmp_path):
    path = save(tmp_path, [row(사업금액=100000000, 입찰마감일="2026-11-30")])
    rows = read_chunks(path)
    assert rows[0]["metadata"]["사업금액"] == 100000000
    assert "사업 금액" not in rows[0]["metadata"]
    report = describe_chunks(path, rows)
    assert report["documents"] == 1 and report["chunks"] == 1
    assert len(report["input_sha256"]) == 64 and report["api_called"] is False


@pytest.mark.parametrize(
    "metadata",
    [
        {"chunk_id": "different.hwp#0"},
        {"chunk_id": "사업.hwp#-1"},
        {"chunk_id": "사업.hwp#가"},
        {"사업금액": "100000000"},
        {"사업 금액": 1.5},
        {"사업금액": True},
        {"입찰마감일": "2026-02-30"},
        {"공개 일자": "20261113"},
        {"chunk_index": -1},
        {"chunk_index": True},
        {"chunk_index": "0"},
        {"chunk_index": 1},
        {"입찰 참여 마감일": "2026-11-30 24:00:00"},
        {"입찰 참여 마감일": "2026-02-30 18:00:00"},
        {"입찰 참여 마감일": "2026-11-30 18:00"},
        {"bad": None},
        {"bad": ["a"]},
    ],
)
def test_bad_metadata_is_rejected_with_line_number(tmp_path, metadata):
    with pytest.raises(ValueError, match="1행"):
        read_chunks(save(tmp_path, [row(**metadata)]))


def test_duplicate_chunk_is_not_silently_dropped(tmp_path):
    with pytest.raises(ValueError, match="2행.*중복"):
        read_chunks(save(tmp_path, [row(), row()]))


@pytest.mark.parametrize("rows", [[], [{"page_content": " ", "metadata": {}}], ["invalid"]])
def test_empty_or_malformed_inputs(tmp_path, rows):
    with pytest.raises(ValueError):
        read_chunks(save(tmp_path, rows))


def test_upstream_chunk_indices_preserve_gaps_and_source_metadata(tmp_path):
    received = [
        {
            "page_content": f"문서의 {index}번째 청크",
            "metadata": {
                "doc_id": "사업.hwp",
                "chunk_index": index,
                "사업 금액": 100000000,
                "입찰 참여 마감일": "2026-11-30 18:00:00",
            },
        }
        for index in (0, 4)
    ]
    path = save(tmp_path, received)
    before = path.read_bytes()
    records = read_chunks(path)
    assert [record["metadata"]["chunk_id"] for record in records] == ["사업.hwp#0", "사업.hwp#4"]
    for record, source in zip(records, received, strict=True):
        assert record["metadata"] == {
            **source["metadata"],
            "chunk_id": f"사업.hwp#{source['metadata']['chunk_index']}",
        }
        assert "chunk_id" not in source["metadata"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", ["2026-11-30", "2026-11-30 00:00:00", "2026-11-30 18:01:59"])
def test_date_and_time_are_preserved(tmp_path, value):
    records = read_chunks(save(tmp_path, [row(**{"입찰 참여 마감일": value})]))
    assert records[0]["metadata"]["입찰 참여 마감일"] == value


def test_duplicate_chunk_across_both_identifier_formats_is_rejected(tmp_path):
    upstream = {"page_content": "다른 본문", "metadata": {"doc_id": "사업.hwp", "chunk_index": 0}}
    with pytest.raises(ValueError, match="2행.*중복"):
        read_chunks(save(tmp_path, [row(), upstream]))


def test_chunk_without_either_identifier_is_rejected(tmp_path):
    received = {"page_content": "본문", "metadata": {"doc_id": "사업.hwp"}}
    with pytest.raises(ValueError, match="1행.*chunk_id 또는 chunk_index"):
        read_chunks(save(tmp_path, [received]))
