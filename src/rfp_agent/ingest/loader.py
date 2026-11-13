
"""[문서 처리] 전체 문서 로딩 및 저장

- data/raw 폴더의 HWP, HWPX, PDF 파일을 탐색한다.
- 파일 형식에 맞는 파서를 실행한다.
- CSV 메타데이터를 문서에 연결한다.
- ParsedDocument 객체로 변환한다.
- JSONL 형식으로 저장하고 다시 불러온다.
"""

import json
import logging
import re
from pathlib import Path

from rfp_agent.ingest.schema import ParsedDocument
from rfp_agent.ingest.hwp import parse_hwp
from rfp_agent.ingest.pdf import parse_pdf
from rfp_agent.ingest.metadata import load_metadata


logger = logging.getLogger(__name__)


def clean_unicode(value):
    """UTF-8 저장이 불가능한 단독 서로게이트 문자를 정리한다."""

    if isinstance(value, str):
        pattern = (
            r"[\ud800-\udbff](?![\udc00-\udfff])"
            r"|(?<![\ud800-\udbff])[\udc00-\udfff]"
        )
        return re.sub(pattern, "\uFFFD", value)

    if isinstance(value, dict):
        return {
            clean_unicode(key): clean_unicode(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [clean_unicode(item) for item in value]

    return value



def load_documents(
    raw_dir: str | Path,
    metadata_file: str | Path
) -> list[ParsedDocument]:
    """원본 폴더의 문서를 파싱하고 메타데이터를 연결한다."""

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {raw_dir}")

    if not raw_dir.is_dir():
        raise NotADirectoryError(f"폴더가 아닙니다: {raw_dir}")

    metadata_map = load_metadata(metadata_file)

    documents = []
    failed_files = []
    skipped_files = []

    source_files = sorted(
        path for path in raw_dir.iterdir()
        if path.is_file()
    )

    total = len(source_files)

    print(f"전체 파일: {total}개", flush=True)
    print("문서 파싱을 시작합니다.", flush=True)

    for index, file_path in enumerate(source_files, start=1):
        extension = file_path.suffix.lower()

        print(
            f"\n[{index}/{total}] 처리 중: {file_path.name}",
            flush=True
        )

        if extension in (".hwp", ".hwpx"):
            parser = parse_hwp
        elif extension == ".pdf":
            parser = parse_pdf
        else:
            skipped_files.append(file_path.name)
            print("  지원하지 않는 파일 형식 - 건너뜀", flush=True)
            continue

        try:
            text = parser(file_path)

            if not text or not text.strip():
                raise ValueError("추출된 텍스트가 없습니다.")

            metadata = metadata_map.get(file_path.name, {}).copy()

            if not metadata:
                logger.warning(
                    "메타데이터를 찾을 수 없습니다: %s",
                    file_path.name
                )

            metadata["파일명"] = file_path.name

            documents.append(
                ParsedDocument(
                    doc_id=file_path.name,
                    text=text,
                    metadata=metadata
                )
            )

            print(
                f"  파싱 성공 | 성공 {len(documents)}건 | 실패 {len(failed_files)}건",
                flush=True
            )

        except Exception as error:
            failed_files.append(file_path.name)

            print(
                f"  파싱 실패: {error} | "
                f"성공 {len(documents)}건 | 실패 {len(failed_files)}건",
                flush=True
            )

            logger.exception("문서 파싱 실패: %s", file_path.name)

    print("\n문서 파싱 완료", flush=True)
    print(f"전체 파일: {total}건")
    print(f"파싱 성공: {len(documents)}건")
    print(f"파싱 실패: {len(failed_files)}건")
    print(f"지원하지 않는 파일: {len(skipped_files)}건")

    if failed_files:
        print("실패 파일 목록:", failed_files)

    return documents



def save_documents(
    docs: list[ParsedDocument],
    path: str | Path
) -> None:
    """ParsedDocument 목록을 JSONL 형식으로 저장한다."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for doc in docs:
            data = {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata
            }

            data = clean_unicode(data)

            file.write(
                json.dumps(data, ensure_ascii=False) + "\n"
            )

    logger.info("문서 저장 완료: %d건 (%s)", len(docs), path)


def read_documents(
    path: str | Path
) -> list[ParsedDocument]:
    """JSONL 파일에서 ParsedDocument 목록을 불러온다."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"저장 파일을 찾을 수 없습니다: {path}")

    documents = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                documents.append(
                    ParsedDocument(
                        doc_id=data["doc_id"],
                        text=data["text"],
                        metadata=data.get("metadata", {})
                    )
                )

            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"JSONL 파일 {line_number}번째 줄 오류: {error}"
                ) from error

    logger.info("문서 로드 완료: %d건 (%s)", len(documents), path)

    return documents
