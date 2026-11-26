"""[프론트·백엔드·평가] 데모 UI (Streamlit)

컨설턴트가 채팅으로 RFP에 대해 묻는 화면입니다. 에이전트는 build_graph() 로 만들어 바로 호출합니다.

실행:  uv run streamlit run app/streamlit_app.py

후속 질문: 브라우저 탭마다 thread_id 를 하나 만들어 매 질문에 넘깁니다. 이전 대화와 대화 중인 공고는
그래프(checkpointer)가 thread_id 별로 기억하므로, 화면에는 새 질문만 넘기면 됩니다.

원문 근거: 답변 아래에서 이번 질문으로 검색한 청크 본문과 출처를 확인할 수 있습니다.
"""

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage

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


def current_search_evidence(messages):
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            break
    else:
        return []

    unique = {}
    for message in messages[index + 1:]:
        if not isinstance(message, ToolMessage) or message.name != "search_rfp":
            continue
        if isinstance(message.artifact, dict):
            for item in message.artifact.get("evidence", []):
                key = (item["source"], item["chunk_id"] or item["text"])
                unique.setdefault(key, item)
    return list(unique.values())


def render_message(role, content, evidence=()):
    with st.chat_message(role):
        st.write(content)
        if evidence:
            with st.expander(f"검색된 원문 근거 보기 ({len(evidence)}개)"):
                for item in evidence:
                    citation = [item["source"]]
                    if item["chunk_id"]:
                        citation.append(f"청크: {item['chunk_id']}")
                    page = item["page_num"]
                    if type(page) is int and page > 0:
                        prefix = "PDF " if item["source"].lower().endswith(".pdf") else ""
                        citation.append(f"{prefix}{page}쪽")
                    st.caption(" | ".join(citation))
                    st.text(item["text"])


if "thread_id" not in st.session_state:
    new_chat()

for message in st.session_state.messages:
    render_message(*message)

if question := st.chat_input("RFP에 대해 물어보세요"):
    render_message("user", question)
    st.session_state.messages.append(("user", question))

    with st.spinner("문서를 찾는 중..."):
        result = get_graph().invoke(
            {"messages": [("user", question)]},
            {"configurable": {"thread_id": st.session_state.thread_id}},
        )
    messages = result["messages"]
    reply = ("assistant", messages[-1].content, current_search_evidence(messages))
    render_message(*reply)
    st.session_state.messages.append(reply)
