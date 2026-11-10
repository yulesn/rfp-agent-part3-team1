"""[문서 처리] PDF 파서

- PDF 1개를 읽어서 텍스트(str)를 돌려주는 함수를 작성합니다.
- 기본은 PyMuPDF(fitz). 표가 깨지면 pdfplumber 등 다른 라이브러리도 비교해 보고,
  비교 결과는 노션 「실험 로그」(분류: 파싱)에 남겨 주세요.
- 신경 쓸 점: 머리글/바닥글·페이지 번호 제거, 표 처리, 페이지 번호를 metadata 로 남길지 여부.
"""

from pathlib import Path


def parse_pdf(path: str | Path) -> str:
    # TODO: fitz.open(path) 로 페이지별 텍스트를 추출해서 합치기
    raise NotImplementedError
