"""[문서 파싱 → 검색] 청킹

- ParsedDocument 를 검색 단위(청크)로 나눠서 LangChain Document 목록으로 돌려줍니다.
- 각 청크의 metadata 에는 원본 문서의 metadata(doc_id, 발주기관, 사업금액 등)를 그대로 복사해 주세요.
  그래야 검색 결과에서 출처를 보여주고 메타데이터로 필터링할 수 있습니다.
- 실험 아이디어: 고정 길이(RecursiveCharacterTextSplitter) vs RFP 목차(장/절) 기준 분할,
  chunk_size / overlap 비교. 설정값은 configs/*.yaml 에서 바꿉니다.
"""

from langchain_core.documents import Document

from rfp_agent.ingest.schema import ParsedDocument


def chunk_documents(docs: list[ParsedDocument], chunk_size: int, chunk_overlap: int) -> list[Document]:
    # TODO: RecursiveCharacterTextSplitter 등으로 분할 후 Document(page_content=..., metadata=...) 생성
    raise NotImplementedError
