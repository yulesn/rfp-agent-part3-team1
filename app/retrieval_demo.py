"""검색 담당 확인 화면. 실행: uv run streamlit run app/retrieval_demo.py"""

import argparse
import os
from pathlib import Path
from time import perf_counter

import streamlit as st

from rfp_agent.config import PROJECT_ROOT, load_config
from rfp_agent.retrieval.retriever import retrieve

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/retrieval_demo.yaml")
args, _ = parser.parse_known_args()
config = load_config(args.config)
db_path = Path(config["paths"]["vectorstore_dir"])
if not db_path.is_absolute():
    db_path = PROJECT_ROOT / db_path
db_exists = (db_path / "chroma.sqlite3").is_file()
key = os.environ.get("OPENAI_API_KEY", "").strip()
key_ready = bool(key and key != "sk-...")

st.set_page_config(page_title="RFP 검색 확인", page_icon="🔎")
st.title("RFP 검색 확인")
st.write("질문을 입력하고, 검색된 원문과 출처를 확인하세요.")
st.caption(
    f"{args.config} · {config['embedding']['model']} · 최대 {config['retrieval']['top_k']}개 검색"
)
if Path(args.config).name == "retrieval_demo.yaml":
    st.info("저장소에 포함된 가상 문서 예제입니다. 검색 결과의 근거 문장을 확인하세요.")
else:
    st.info("선택한 설정의 DB를 검색합니다. 검색 결과의 근거 문장을 확인하세요.")
if not db_exists:
    st.warning("벡터DB가 없습니다. docs/retrieval.md의 예제 DB 생성 순서를 먼저 실행해 주세요.")
if not key_ready:
    st.warning("프로젝트 .env에 OPENAI_API_KEY를 설정해 주세요. 키 값은 화면에 입력하지 않습니다.")

with st.form("retrieval_question"):
    question = st.text_input(
        "질문", placeholder="한빛시청 도서관 통합 시스템의 사업 예산과 기간은?"
    )
    submitted = st.form_submit_button(
        "검색", disabled=not (db_exists and key_ready), type="primary"
    )
st.caption("검색할 때 질문 임베딩 API를 호출합니다. 답변 생성 LLM은 호출하지 않습니다.")

if submitted:
    st.session_state.pop("search_result", None)
    if not question.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        started = perf_counter()
        try:
            with st.spinner("관련 문장을 찾는 중..."):
                # 화면은 입력·표시만 맡고, 검색은 공통 함수가 수행합니다.
                documents = retrieve(question, config)
            st.session_state["search_result"] = (question, documents, perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 -- 화면 경계에서 외부 오류 본문을 숨깁니다.
            # API 오류 본문에는 키 일부가 포함될 수 있어 종류와 상태만 표시합니다.
            st.error("검색하지 못했습니다. DB 상태와 API 연결을 확인해 주세요.")
            st.caption(
                f"오류 종류: {type(exc).__name__} · HTTP 상태: {getattr(exc, 'status_code', None)}"
            )

if "search_result" in st.session_state:
    previous_question, documents, elapsed = st.session_state["search_result"]
    st.subheader("검색 결과")
    st.caption(f"질문: {previous_question}")
    st.write(f"관련 청크 {len(documents)}개 · {elapsed:.2f}초")
    if not documents:
        st.info("검색된 문장이 없습니다.")
    for rank, document in enumerate(documents, 1):
        with st.expander(f"{rank}. {document.metadata['doc_id']}", expanded=rank == 1):
            st.caption(f"청크 ID: {document.metadata['chunk_id']}")
            if document.metadata.get("page_num") not in (None, ""):
                st.caption(f"페이지: {document.metadata['page_num']}")
            st.write(document.page_content)
