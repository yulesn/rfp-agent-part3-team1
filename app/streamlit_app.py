"""[프론트·백엔드·평가] 데모 UI (Streamlit)

컨설턴트가 채팅으로 RFP에 대해 묻는 화면입니다. 에이전트는 build_graph() 로 만들어 바로 호출합니다.

실행:  uv run streamlit run app/streamlit_app.py

추가하면 좋은 것: 답변의 출처(문서명) 표시, 도구 호출 과정 펼쳐보기, 예시 질문 버튼, 이전 대화 이어가기
"""

import httpx
import streamlit as st

st.set_page_config(page_title="RFP 어시스턴트", page_icon="🏛️")
st.title("🏛️ 입찰메이트 RFP 어시스턴트")
st.caption(
    "나라장터 제안요청서(RFP)의 핵심 요구사항, 예산, 출처를 빠르게 찾아드립니다."
)

# 백엔드 서버 주소 (터미널 1번에서 켜둔 FastAPI 서버)
API_URL = "http://localhost:8000/ask"

# 대화 기록 저장 공간
if "messages" not in st.session_state:
    st.session_state.messages = []

# 왼쪽 사이드바: 예시 질문 버튼 & 대화 초기화
with st.sidebar:
    st.header("💡 예시 질문")
    st.write("버튼을 누르면 바로 질문이 입력됩니다.")
    example_questions = [
        "국민연금공단 이러닝시스템 사업 요구사항 정리해 줘.",
        "기초과학연구원 극저온시스템 사업 예산이 얼마야?",
        "한국원자력연구원 선량 평가 시스템 사업 목적 알려줘.",
    ]
    selected_example = None
    for q in example_questions:
        if st.button(q, use_container_width=True):
            selected_example = q

    st.divider()
    if st.button("🗑️ 대화 내용 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 이전 대화 기록 화면에 보여주기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 참고 문서 및 근거 출처 보기"):
                for src in msg["sources"]:
                    doc_id = src.get("doc_id", "문서 ID 없음")
                    title = src.get("title", "")
                    snippet = src.get("snippet", "")
                    st.markdown(f"**• {title}** (`{doc_id}`)")
                    if snippet:
                        st.caption(f"“{snippet}”")

# 채팅 입력창 또는 예시 버튼 클릭 처리
user_input = st.chat_input("RFP에 대해 물어보세요...")
question = selected_example or user_input

if question:
    # 1. 내 질문 화면에 띄우기
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 2. 백엔드 서버(/ask)에 질문 보내고 답변 받아오기
    with st.chat_message("assistant"):
        with st.spinner("문서를 찾는 중..."):
            try:
                response = httpx.post(
                    API_URL, json={"question": question}, timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "답변을 생성하지 못했습니다.")
                sources = data.get("sources", [])

                st.write(answer)
                if sources:
                    with st.expander(
                        "📄 참고 문서 및 근거 출처 보기", expanded=True
                    ):
                        for src in sources:
                            doc_id = src.get("doc_id", "문서 ID 없음")
                            title = src.get("title", "")
                            snippet = src.get("snippet", "")
                            st.markdown(f"**• {title}** (`{doc_id}`)")
                            if snippet:
                                st.caption(f"“{snippet}”")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except httpx.ConnectError:
                st.error(
                    "⚠️ 백엔드 서버(localhost:8000)에 연결할 수 없습니다. 터미널 1번에서 uvicorn 서버가 켜져 있는지 확인해 주세요!"
                )
            except Exception as e:
                st.error(f"⚠️ 오류가 발생했습니다: {e}")