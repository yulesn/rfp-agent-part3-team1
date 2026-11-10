"""[프론트·백엔드·평가] API 요청·응답 형식

- UI(Streamlit)와 평가 스크립트가 API 서버와 주고받는 데이터 형식을 정의합니다.
- 답변 근거·출처를 보여주는 게 기본 기능이라, 응답에 출처 목록을 꼭 포함해 주세요.
"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    # TODO: 대화를 이어가려면 thread_id(대화 ID) 추가


class Source(BaseModel):
    doc_id: str          # 파일명 등 문서 ID
    title: str = ""      # 사업명
    snippet: str = ""    # 근거가 된 청크 일부


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = []
