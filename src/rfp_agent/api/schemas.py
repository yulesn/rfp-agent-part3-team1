"""[프론트·백엔드·평가] API 요청·응답 형식

- UI(Streamlit)와 평가 스크립트가 API 서버와 주고받는 데이터 형식을 정의합니다.
- 답변 근거·출처를 보여주는 게 기본 기능이라, 응답에 출처 목록을 꼭 포함해 주세요.
"""

from typing import List
from pydantic import BaseModel

# 프론트엔드에서 보낼 질문 모양
class AskRequest(BaseModel):
    question: str

# 출처 1개에 들어갈 정보 3가지
class SourceItem(BaseModel):
    doc_id: str
    title: str
    snippet: str

# 백엔드가 돌려줄 최종 답변 모양
class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
