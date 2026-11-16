
"""[문서 처리] PDF 파서

- Docling을 사용해 PDF 문서를 파싱한다.
- 문서의 읽기 순서를 유지하면서 텍스트와 표를 추출한다.
- 표는 Markdown 형식으로 변환한다.
"""

from pathlib import Path

from docling.document_converter import DocumentConverter


pdf_converter = DocumentConverter()


def parse_pdf(path: str | Path) -> str:
    """PDF 파일을 읽어 본문 텍스트를 반환한다."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    conversion = pdf_converter.convert(str(path))
    doc = conversion.document

    # Docling이 문서 구조와 읽기 순서를 반영해 Markdown으로 변환
    text = doc.export_to_markdown()

    if not text or not text.strip():
        raise ValueError(f"PDF에서 텍스트를 추출하지 못했습니다: {path}")

    return text.strip()
