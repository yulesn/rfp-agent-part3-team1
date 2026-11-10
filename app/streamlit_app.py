"""[프론트·백엔드·평가] 데모 UI (Streamlit)

컨설턴트가 채팅으로 RFP에 대해 묻는 화면입니다. 에이전트는 build_graph() 로 만들어 바로 호출합니다.

실행:  uv run streamlit run app/streamlit_app.py

추가하면 좋은 것: 답변의 출처(문서명) 표시, 도구 호출 과정 펼쳐보기, 예시 질문 버튼, 이전 대화 이어가기
"""

import streamlit as st

from rfp_agent.agent.graph import build_graph
from rfp_agent.config import load_config

st.set_page_config(page_title="RFP 어시스턴트", page_icon="🏛️")
st.title("🏛️ RFP 어시스턴트")


@st.cache_resource  # 앱이 다시 그려질 때마다 그래프를 새로 만들지 않도록 캐시
def get_graph():
    return build_graph(load_config())


if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    st.chat_message(role).write(content)

if question := st.chat_input("RFP에 대해 물어보세요"):
    st.chat_message("user").write(question)
    st.session_state.messages.append(("user", question))

    with st.spinner("문서를 찾는 중..."):
        # TODO: 이전 대화를 이어가려면 st.session_state.messages 전체를 넘기기
        result = get_graph().invoke({"messages": [("user", question)]})
    answer = result["messages"][-1].content

    st.chat_message("assistant").write(answer)
    st.session_state.messages.append(("assistant", answer))
