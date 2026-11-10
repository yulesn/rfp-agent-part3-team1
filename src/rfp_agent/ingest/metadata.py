"""[문서 처리] 메타데이터 불러오기

- 함께 제공받은 메타데이터 파일(csv 등)을 읽어서 파일명 → 메타데이터 dict 로 매칭하는 함수를 작성합니다.
- 여기서 정리한 메타데이터(발주기관, 사업금액, 마감일 등)는 청크에 함께 저장되어
  에이전트의 메타데이터 필터 검색 도구(agent/tools/filter_metadata.py)에서 사용됩니다.
- 신경 쓸 점: 금액은 숫자(int)로, 날짜는 "YYYY-MM-DD" 문자열로 통일하면 필터링이 쉬워집니다.
"""

from pathlib import Path


def load_metadata(path: str | Path) -> dict[str, dict]:
    # TODO: pandas.read_csv(path) 후 {파일명: {컬럼: 값, ...}} 형태로 변환
    raise NotImplementedError
