"""[문서 파싱] 파싱 결과 형식 (팀 공통 약속)

- 파싱 파트와 검색 파트가 주고받는 데이터 형식입니다. 모든 파서는 이 형식으로 결과를 돌려줍니다.
- 필드를 바꾸거나 추가할 때는 팀에 먼저 공유하고 노션 「의사결정 로그」에 남겨 주세요.
"""

from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    doc_id: str          # 파일 이름 등 문서를 구분하는 고유 ID
    text: str            # 문서 전체 텍스트 (표는 마크다운 표 등 텍스트로 변환)
    metadata: dict = field(default_factory=dict)
    # metadata 예시: {"source": "파일명.hwp", "file_type": "hwp", "발주기관": "...",
    #                "사업명": "...", "사업금액": 0, "공고일": "...", "입찰마감일": "..."}
    # → 메타데이터 키 이름은 제공받은 메타데이터 파일 컬럼에 맞춰 정하세요.
