"""[프론트·백엔드·평가] API 요청·응답 형식"""

from __future__ import annotations

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    thread_id: str | None = None


class Source(BaseModel):
    doc_id: str                        # 파일명 (source)
    title: str = ""                    # 사업명 (선택)
    snippet: str = ""                  # 청크 본문 (text)
    chunk_id: str | int | None = None  # 청크 ID
    page_num: int | str | None = None  # PDF 쪽수


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = []