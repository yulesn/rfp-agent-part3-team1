"""[프론트·백엔드·평가] 데모 UI (Streamlit)

컨설턴트가 채팅으로 RFP에 대해 묻는 화면입니다. 에이전트는 build_graph() 로 만들어 바로 호출합니다.

실행:  uv run streamlit run app/streamlit_app.py

후속 질문: 브라우저 탭마다 thread_id 를 하나 만들어 매 질문에 넘깁니다. 이전 대화와 대화 중인 공고는
그래프(checkpointer)가 thread_id 별로 기억하므로, 화면에는 새 질문만 넘기면 됩니다.

추가하면 좋은 것: 답변의 출처(문서명) 표시, 도구 호출 과정 펼쳐보기, 예시 질문 버튼
"""

import uuid

import streamlit as st

from rfp_agent.agent.graph import build_graph, new_memory
from rfp_agent.config import load_config

st.set_page_config(page_title="RFP 어시스턴트", page_icon="🏛️")
st.title("🏛️ RFP 어시스턴트")


@st.cache_resource  # 앱이 다시 그려질 때마다 그래프(대화 기록 포함)를 새로 만들지 않도록 캐시
def get_graph():
    return build_graph(load_config(), checkpointer=new_memory())


def new_chat() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


if "thread_id" not in st.session_state:
    new_chat()

for role, content in st.session_state.messages:
    st.chat_message(role).write(content)

if question := st.chat_input("RFP에 대해 물어보세요"):
    st.chat_message("user").write(question)
    st.session_state.messages.append(("user", question))

    with st.spinner("문서를 찾는 중..."):
        result = get_graph().invoke(
            {"messages": [("user", question)]},
            {"configurable": {"thread_id": st.session_state.thread_id}},
        )
    answer = result["messages"][-1].content

    st.chat_message("assistant").write(answer)
    st.session_state.messages.append(("assistant", answer))
