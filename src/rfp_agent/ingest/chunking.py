"""[문서 처리] 청킹

- ParsedDocument 를 검색 단위(청크)로 나눠서 LangChain Document 목록으로 돌려줍니다.
- 각 청크의 metadata 에는 원본 문서의 metadata(doc_id, 발주기관, 사업금액 등)를 그대로 복사해 주세요.
  그래야 검색 결과에서 출처를 보여주고 메타데이터로 필터링할 수 있습니다.
- 실험 아이디어: 고정 길이(RecursiveCharacterTextSplitter) vs RFP 목차(장/절) 기준 분할,
  chunk_size / overlap 비교. 설정값은 configs/*.yaml 에서 바꿉니다.
- 각 청크에 원본 문서 메타데이터를 복사한다.
- PDF는 페이지별로 청킹하여 페이지 번호를 보존한다.
- HWP/HWPX는 기존 방식대로 전체 텍스트를 청킹한다.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rfp_agent.ingest.schema import ParsedDocument


def chunk_documents(
    docs: list[ParsedDocument],
    chunk_size: int,
    chunk_overlap: int
) -> list[Document]:
    """파싱된 문서를 지정한 크기의 청크로 분할한다."""

    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap은 0 이상이고 chunk_size보다 작아야 합니다."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for doc in docs:
        if not doc.text or not doc.text.strip():
            continue

        # PDF: 페이지별로 분할하여 페이지 번호 보존
        if doc.pages:
            text_units = [
                (page["text"], page["page_num"])
                for page in doc.pages
                if page.get("text") and page["text"].strip()
            ]

        # HWP/HWPX: 기존 전체 텍스트 분할 방식 유지
        else:
            text_units = [(doc.text, None)]

        chunk_index = 0

        for source_text, page_num in text_units:
            split_texts = splitter.split_text(source_text)

            for text in split_texts:
                normalized = text.strip()

                # 의미 없는 Markdown 표 구분선 제거
                if re.fullmatch(
                    r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$",
                    normalized
                ):
                    continue

                # 단독 파이프 제거
                if normalized == "|":
                    continue

                metadata = {
                    **doc.metadata,
                    "doc_id": doc.doc_id,
                    "chunk_index": chunk_index
                }

                # PDF 청크에만 페이지 번호 추가
                if page_num is not None:
                    metadata["page_num"] = page_num

                chunks.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )

                chunk_index += 1

    return chunks
