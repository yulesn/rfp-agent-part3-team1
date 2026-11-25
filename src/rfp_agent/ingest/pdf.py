"""[문서 처리] PDF 파서

- Docling을 사용해 PDF 문서를 파싱한다.
- 문서의 읽기 순서를 유지하면서 텍스트와 표를 추출한다.
- 표는 Markdown 형식으로 변환한다.
- 페이지 번호를 보존하여 RAG 답변 출처에 활용한다.
"""

from pathlib import Path

from docling.document_converter import DocumentConverter


pdf_converter = DocumentConverter()


def parse_pdf(path: str | Path) -> tuple[str, list[dict]]:
    """PDF 파일의 전체 텍스트와 페이지별 텍스트를 반환한다."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    conversion = pdf_converter.convert(str(path))
    doc = conversion.document

    # 문서 전체 텍스트 추출
    text = doc.export_to_markdown()

    if not text or not text.strip():
        raise ValueError(f"PDF에서 텍스트를 추출하지 못했습니다: {path}")

    # 페이지별 텍스트 및 실제 페이지 번호 저장
    pages = []

    for page_no in doc.pages.keys():
        page_text = doc.export_to_markdown(page_no=page_no).strip()

        if page_text:
            pages.append(
                {
                    "page_num": page_no,
                    "text": page_text,
                }
            )

    return text.strip(), pages