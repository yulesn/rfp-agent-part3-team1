"""[문서 처리] 전체 문서 로딩

- data/raw 폴더의 모든 파일을 확장자에 맞는 파서(hwp.py, pdf.py)로 읽고,
  메타데이터(metadata.py)를 붙여서 ParsedDocument 목록을 돌려줍니다.
- 결과는 data/processed/documents.jsonl 로 저장해 두면 매번 다시 파싱하지 않아도 됩니다.
- 파싱에 실패한 파일은 건너뛰고 로그로 남겨 주세요 (100건 중 몇 건이 실패했는지도 보고서 재료).
"""

from pathlib import Path

from rfp_agent.ingest.schema import ParsedDocument


def load_documents(raw_dir: str | Path, metadata_file: str | Path) -> list[ParsedDocument]:
    # TODO: raw_dir 의 파일을 돌면서 .hwp → parse_hwp, .pdf → parse_pdf 호출
    #       load_metadata 결과를 doc.metadata 에 합치기
    raise NotImplementedError


def save_documents(docs: list[ParsedDocument], path: str | Path) -> None:
    # TODO: jsonl 로 저장 (한 줄에 문서 하나, ensure_ascii=False)
    raise NotImplementedError


def read_documents(path: str | Path) -> list[ParsedDocument]:
    # TODO: save_documents 로 저장한 jsonl 을 다시 읽기
    raise NotImplementedError
