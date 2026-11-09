"""[문서 파싱] HWP 파서

- HWP 1개를 읽어서 텍스트(str)를 돌려주는 함수를 작성합니다.
- HWP는 OLE 구조라서 olefile 로 "BodyText/Section*" 스트림을 열고 zlib 압축을 풀어 텍스트를 꺼낼 수 있습니다.
- 다른 방법: pyhwp(hwp5txt) 사용, LibreOffice 로 PDF 변환 후 pdf 파서 재사용 등.
  어떤 방법이 표·문단을 잘 살리는지 비교해 보고 노션 「실험 로그」(분류: 파싱)에 기록해 주세요.
"""

from pathlib import Path


def parse_hwp(path: str | Path) -> str:
    # TODO: olefile.OleFileIO(path) → BodyText 섹션 읽기 → zlib.decompress(..., -15) → 텍스트 레코드 추출
    raise NotImplementedError
