"""[프론트·백엔드·평가] 반응형 대화형 데모 UI (Streamlit)

후속 질문: 브라우저 탭마다 thread_id 를 하나 만들어 매 질문에 넘깁니다. 이전 대화와 대화 중인 공고는
그래프(checkpointer)가 thread_id 별로 기억하므로, 화면에는 새 질문만 넘기면 됩니다.
원문 근거: 답변 아래에서 이번 질문으로 검색한 청크 본문과 출처를 확인할 수 있습니다.

실행: uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import random
import re
import time
from uuid import uuid4

import httpx
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="#입찰메이트 · RFP 어시스턴트",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "http://localhost:8000/ask"
HEALTH_URL = "http://localhost:8000/health"

FAQ_POOL: list[str] = [
    "서민금융진흥원 채팅 상담시스템 구축 사업의 사업예산과 사업기간은?",
    "서민금융진흥원 채팅 상담시스템 구축 사업의 무상 하자보수 기간은 얼마인가요?",
    "기초과학연구원 2025년도 중이온가속기용 극저온시스템 운전 용역의 과업기간은 언제까지인가요?",
    "기초과학연구원 극저온시스템 운전 용역의 주요 요구사항을 요약해 주세요.",
    "국민연금공단 2024년 이러닝시스템 운영 용역의 총 사업예산과 사업기간을 알려주세요.",
    "국민연금공단 2024년 이러닝시스템 운영 용역에서 교육과정 운영 예산과 시스템 임대 예산 차이는 얼마인가요?",
    "경기도사회서비스원 2024년 통합사회정보시스템 운영지원 사업 예산과 부가세 포함 여부는?",
    "경기도사회서비스원 통합사회정보시스템 운영지원 제안사 평가에서 기술능력과 가격평가 비중은?",
    "부산관광공사 경영정보시스템 기능개선 사업에 공동수급체로 입찰할 때 구성원 수와 최소 지분율 조건은?",
    "부산관광공사 경영정보시스템 기능개선 사업의 하자보수 기간은 어떻게 되나요?",
    "고려대학교 차세대 포털·학사 정보시스템이 지원해야 하는 평균 및 폭증 시 동시 사용자 수는?",
    "고려대학교 차세대 포털·학사 정보시스템 구축 사업의 제안 설명회 발표 시간과 발표자는 누구인가요?",
    "고려대학교 차세대 포털·학사, 부산관광공사, 서민금융진흥원 사업의 무상 하자보수 기간을 비교해 주세요.",
    "전북대학교 JST 공유대학(원) xAPI 기반 LRS시스템 구축 사업의 목적과 주요 내용을 요약해 주세요.",
    "전북대학교 JST 공유대학(원) xAPI 기반 LRS시스템 구축 사업의 사업기간은 며칠인가요?",
    "한국전기안전공사 전기안전 관제시스템 보안 모듈 개발 용역의 사업 기간은 어떻게 되나요?",
    "한국전기안전공사 기존 운영시스템 현황에서 WEB, 통신서버, WAS, DB는 각각 몇 대인가요?",
    "한국장애인문화예술원 2024년 장애인문화예술정보시스템 이음 운영 사업의 총 소요예산은 얼마인가요?",
    "경희대학교 산학협력단 정보시스템 운영 용역업체 선정 공고의 주요 과업 내용을 요약해 주세요.",
    "경기도 안양시 호계체육관 배드민턴장 및 탁구장 예약시스템 구축 용역의 핵심 요구사항은 무엇인가요?",
    "경상북도 봉화군 재난통합관리시스템 고도화 사업의 추진 목적과 주요 내용을 알려주세요.",
    "서민금융진흥원 채팅 상담시스템 구축 사업의 계약 체결 후 완료 기한은 몇 개월 이내인가요?",
    "전기안전 관제시스템 보안 모듈 개발 사업과 JST 공유대학 LRS시스템 구축 사업의 사업기간 차이는 며칠인가요?",
    "고려대학교 차세대 포털·학사 정보시스템 구축 사업의 무상유지보수 기간은 언제부터 몇 개월인가요?",
]


def new_chat() -> None:
    """대화 내역을 비우고 새 thread_id를 발급합니다."""
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid4())
    st.session_state.pop("pending_question", None)


def _check_backend_health() -> tuple[bool, str]:
    """백엔드 /health 상태를 조회해 (연결 여부, 활성 LLM 모델명)을 반환합니다."""
    try:
        res = httpx.get(HEALTH_URL, timeout=1.5)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "ok":
                model_name = str(data.get("model", "")).strip()
                return True, model_name
    except Exception:
        pass
    return False, ""


def _clean_html(raw_html: str) -> str:
    """마크다운이 들여쓰기를 코드 블록으로 오인하지 않도록 앞뒤 공백을 제거합니다."""
    return "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())


def _scroll_to_bottom() -> None:
    """질문 입력/클릭 즉시 및 DOM 변경 시마다 창 맨 아래로 스크롤합니다."""
    nonce = time.time_ns()
    components.html(
        f"""
        <script>
        // nonce: {nonce}
        (function() {{
            const parentWin = window.parent;
            const doc = parentWin.document;

            const scrollNow = () => {{
                const targets = [
                    doc.querySelector('[data-testid="stAppViewContainer"]'),
                    doc.querySelector('[data-testid="stMain"]'),
                    doc.querySelector('section.main'),
                    doc.documentElement,
                    doc.body
                ];
                targets.forEach(el => {{
                    if (el) {{
                        el.scrollTop = el.scrollHeight;
                        el.scrollTo({{ top: el.scrollHeight, behavior: 'smooth' }});
                    }}
                }});
                parentWin.scrollTo({{ top: doc.body.scrollHeight, behavior: 'smooth' }});
            }};

            scrollNow();
            requestAnimationFrame(scrollNow);
            setTimeout(scrollNow, 60);
            setTimeout(scrollNow, 180);
            setTimeout(scrollNow, 360);

            if (!parentWin.__rfpScrollObserver) {{
                const mainContainer = doc.querySelector('[data-testid="stMain"]') || doc.body;
                const observer = new MutationObserver(() => {{
                    if (parentWin.__rfpAutoScrollActive) {{
                        scrollNow();
                    }}
                }});
                observer.observe(mainContainer, {{ childList: true, subtree: true }});
                parentWin.__rfpScrollObserver = observer;
            }}
            parentWin.__rfpAutoScrollActive = true;
            setTimeout(() => {{ parentWin.__rfpAutoScrollActive = false; }}, 1200);
        }})();
        </script>
        """,
        height=0,
    )


CUSTOM_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

:root {
    --bg-canvas: #F2F2F2;
    --ink-dark: #111111;
    --bubble-yellow: #FFA800;
    --bubble-orange: #FF5500;
    --bubble-blue: #0077FF;
    --bubble-cyan: #38C6F4;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-canvas) !important;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
    color: #111111 !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

iframe[height="0"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.block-container {
    max-width: 1260px !important;
    padding-top: 1.1rem !important;
    padding-bottom: 7.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2),
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
    position: sticky !important;
    top: 1.1rem !important;
    z-index: 30;
    background: rgba(255, 255, 255, 0.96) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: none !important;
    outline: none !important;
    border-radius: 26px !important;
    padding: 20px 18px !important;
    box-shadow: 0 12px 32px rgba(17, 17, 17, 0.07) !important;
    box-sizing: border-box !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) > div > [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) > div > [data-testid="stVerticalBlock"] {
    gap: 0.72rem !important;
}

div.stButton {
    width: 100% !important;
    max-width: 100% !important;
}

div.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"],
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    box-sizing: border-box !important;
    background-color: #F4F5F7 !important;
    background: #F4F5F7 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 13px !important;
    color: #111111 !important;
    font-weight: 700 !important;
    font-size: 0.83rem !important;
    line-height: 1.45 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    word-break: keep-all !important;
    overflow-wrap: break-word !important;
    padding: 0.68rem 0.9rem !important;
    margin: 0 !important;
    transition: background-color 0.18s ease, transform 0.18s ease !important;
}

div.stButton > button:hover,
div.stButton > button:focus,
div.stButton > button:active,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background-color: #E5E7EB !important;
    background: #E5E7EB !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #111111 !important;
    transform: translateY(-1px);
}

div.stButton > button > div,
div.stButton > button [data-testid="stMarkdownContainer"],
div.stButton > button p,
div.stButton > button span {
    color: #111111 !important;
    font-weight: 700 !important;
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    white-space: normal !important;
    word-break: keep-all !important;
    overflow-wrap: break-word !important;
    text-overflow: clip !important;
    overflow: visible !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"],
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:first-of-type div.stButton,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:first-of-type div.stButton {
    width: auto !important;
    display: inline-flex !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:first-of-type div.stButton > button,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:first-of-type div.stButton > button {
    width: 32px !important;
    min-width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    border: none !important;
    outline: none !important;
    background-color: #F3F4F6 !important;
    background: #F3F4F6 !important;
    color: #111111 !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    transition: transform 0.3s ease, background-color 0.18s ease !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:first-of-type div.stButton > button:hover,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:first-of-type div.stButton > button:hover {
    background-color: #E5E7EB !important;
    background: #E5E7EB !important;
    color: #111111 !important;
    transform: scale(1.1) !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:first-of-type div.stButton > button p,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:first-of-type div.stButton > button p {
    white-space: nowrap !important;
    line-height: 1 !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type {
    margin-top: 6px !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type div.stButton,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type div.stButton {
    width: auto !important;
    display: flex !important;
    justify-content: center !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type div.stButton > button,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type div.stButton > button {
    width: auto !important;
    min-width: 152px !important;
    padding: 0.56rem 1.4rem !important;
    border-radius: 999px !important;
    border: none !important;
    outline: none !important;
    background-color: #F3F4F6 !important;
    background: #F3F4F6 !important;
    color: #111111 !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type div.stButton > button > div,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type div.stButton > button > div,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type div.stButton > button p,
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stHorizontalBlock"]:last-of-type div.stButton > button p {
    width: auto !important;
    white-space: nowrap !important;
    text-align: center !important;
}

.conv-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 1.1rem;
    padding: 0 4px;
}

.conv-title {
    font-size: clamp(1.9rem, 4vw, 2.8rem);
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #111111 !important;
    line-height: 1.05;
    margin: 0;
}

.conv-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #FFFFFF;
    color: #111111 !important;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 7px 14px;
    border: none !important;
    border-radius: 999px;
    box-shadow: 0 4px 12px rgba(17, 17, 17, 0.06);
    white-space: nowrap;
}

.conv-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.conv-dot.online {
    background: #00C853;
    animation: pulseDot 1.8s infinite ease-in-out;
}

.conv-dot.offline {
    background: #FF3B30;
}

.hero-stack {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 20px;
    width: 100%;
    height: calc(100vh - 205px);
    min-height: 560px;
    padding-bottom: 16px;
    overflow: visible !important;
}

.bubble-anim-wrap {
    flex: 1 1 25%;
    height: 25%;
    display: flex;
    width: 100%;
    animation: bubblePopIn 0.55s cubic-bezier(0.22, 1.35, 0.36, 1) both;
    overflow: visible !important;
}

.chunky-bubble {
    position: relative;
    width: 88%;
    height: 100%;
    padding: 18px 28px;
    border: none !important;
    color: #111111 !important;
    filter: drop-shadow(0 8px 18px rgba(17, 17, 17, 0.11));
    transition: transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1), filter 0.28s ease !important;
    word-break: keep-all;
    overflow-wrap: break-word;
    cursor: pointer;
    will-change: transform;
}

.zigzag-left {
    margin-left: 0;
    margin-right: auto;
}

.zigzag-right {
    margin-left: auto;
    margin-right: 0;
}

.chunky-bubble:hover {
    transform: translateY(-4px) scale(1.03) !important;
    filter: drop-shadow(0 16px 28px rgba(17, 17, 17, 0.18)) !important;
    z-index: 25 !important;
}

.theme-yellow {
    background: var(--bubble-yellow);
    border-radius: 32px;
    z-index: 4;
}
.theme-yellow::after {
    content: "";
    position: absolute;
    bottom: -16px;
    right: 26%;
    width: 28px;
    height: 18px;
    background: var(--bubble-yellow);
    clip-path: polygon(0 0, 72% 0, 100% 100%);
}

.theme-orange {
    background: var(--bubble-orange);
    border-radius: 52px;
    z-index: 3;
}
.theme-orange::after {
    content: "";
    position: absolute;
    bottom: -16px;
    left: 26%;
    width: 28px;
    height: 18px;
    background: var(--bubble-orange);
    clip-path: polygon(28% 0, 100% 0, 0 100%);
}

.theme-blue {
    background: var(--bubble-blue);
    border-radius: 28px;
    z-index: 2;
}
.theme-blue::after {
    content: "";
    position: absolute;
    bottom: -16px;
    right: 26%;
    width: 28px;
    height: 18px;
    background: var(--bubble-blue);
    clip-path: polygon(0 0, 72% 0, 100% 100%);
}

.theme-cyan {
    background: var(--bubble-cyan);
    border-radius: 18px;
    z-index: 1;
}
.theme-cyan::after {
    content: "";
    position: absolute;
    bottom: -16px;
    left: 26%;
    width: 28px;
    height: 18px;
    background: var(--bubble-cyan);
    clip-path: polygon(28% 0, 100% 0, 0 100%);
}

.hero-bubble {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    gap: 8px;
}

.hero-hint {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: clamp(0.95rem, 1.65vw, 1.06rem);
    font-weight: 700;
    line-height: 1.45;
    color: #111111 !important;
}

.face-top, .face-bottom {
    display: flex;
    justify-content: center;
    align-items: center;
    pointer-events: none;
}

.cute-face {
    animation: faceFloat 3.2s ease-in-out infinite;
}

.chat-bubble {
    padding: 18px 22px;
    margin-bottom: 18px;
    line-height: 1.65;
    border: none !important;
    outline: none !important;
    word-break: keep-all;
    overflow-wrap: break-word;
    animation: chatFadeIn 0.3s ease-out both;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.user-bubble {
    max-width: 80%;
    margin-left: auto;
    margin-right: 0;
    background: #0077FF;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 22px 22px 6px 22px;
    box-shadow: 0 6px 18px rgba(0, 119, 255, 0.18);
}

.assistant-bubble {
    max-width: 88%;
    margin-left: 0;
    margin-right: auto;
    background: #FFFFFF;
    color: #111111 !important;
    border: none !important;
    border-radius: 22px 22px 22px 6px;
    box-shadow: 0 6px 18px rgba(17, 17, 17, 0.05);
}

.chat-meta {
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 6px;
    opacity: 0.75;
}

.user-bubble .chat-meta,
.user-bubble .chat-body {
    color: #FFFFFF !important;
}

.assistant-bubble .chat-meta,
.assistant-bubble .chat-body {
    color: #111111 !important;
}

.chat-body {
    font-size: 1.0rem;
    font-weight: 500;
}

.typing-dots {
    display: inline-flex;
    gap: 6px;
    align-items: center;
    padding: 4px 0;
}
.typing-dots span {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #111111;
    animation: typingBounce 1.1s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.16s; }
.typing-dots span:nth-child(3) { animation-delay: 0.32s; }

.faq-box-title {
    font-size: 1.02rem;
    font-weight: 900;
    color: #111111 !important;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    line-height: 32px;
    margin: 0 0 12px 36px;
}

/* 검색된 원문 근거 보기 토글 */
[data-testid="stExpander"] {
    width: fit-content !important;
    max-width: 88% !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    margin-top: -8px !important;
    margin-bottom: 20px !important;
}

[data-testid="stExpander"] details {
    width: fit-content !important;
    max-width: 100% !important;
    border: none !important;
    outline: none !important;
    border-radius: 14px !important;
    background: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(17, 17, 17, 0.06) !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] details > summary {
    width: fit-content !important;
    list-style: none !important;
    border: none !important;
    outline: none !important;
    padding: 9px 16px !important;
    border-radius: 14px !important;
    background-color: #FFFFFF !important;
    color: #111111 !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    transition: background-color 0.18s ease !important;
    cursor: pointer !important;
}

[data-testid="stExpander"] details > summary p,
[data-testid="stExpander"] details > summary span,
[data-testid="stExpander"] details > summary div {
    color: #111111 !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] details > summary svg,
[data-testid="stExpander"] details > summary svg path {
    color: #111111 !important;
    fill: #111111 !important;
}

[data-testid="stExpander"] details > summary:hover {
    background-color: #F3F4F6 !important;
    color: #111111 !important;
}

[data-testid="stExpander"] details[open] > summary {
    width: 100% !important;
    border-radius: 14px 14px 0 0 !important;
    background-color: #F4F5F7 !important;
    color: #111111 !important;
}

[data-testid="stExpanderDetails"] {
    border: none !important;
    outline: none !important;
    background-color: #FFFFFF !important;
    padding: 14px 18px 16px 18px !important;
    color: #111111 !important;
}

[data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"],
[data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"] p {
    color: #374151 !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}

[data-testid="stExpanderDetails"] pre,
[data-testid="stExpanderDetails"] code {
    background-color: #F8F9FA !important;
    color: #111111 !important;
    border: none !important;
    border-radius: 10px !important;
    white-space: pre-wrap !important;
    word-break: keep-all !important;
}

[data-testid="stChatInput"] {
    max-width: 1220px;
    margin: 0 auto;
}

[data-testid="stChatInput"] * {
    border: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] > div {
    min-height: 68px !important;
    border-radius: 26px !important;
    border: none !important;
    outline: none !important;
    background: #FFFFFF !important;
    padding: 8px 12px 8px 20px !important;
    box-shadow: 0 10px 28px rgba(17, 17, 17, 0.09) !important;
    display: flex !important;
    align-items: center !important;
}

[data-testid="stChatInputTextArea"] {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #111111 !important;
    text-align: left !important;
    line-height: 1.5 !important;
    padding-top: 12px !important;
    padding-bottom: 12px !important;
    min-height: 48px !important;
    border: none !important;
    outline: none !important;
}

[data-testid="stChatInputTextArea"]::placeholder {
    color: #111111 !important;
    opacity: 0.48 !important;
    text-align: left !important;
    font-weight: 600 !important;
}

[data-testid="stChatInputSubmitButton"] {
    background-color: #111111 !important;
    color: #FFFFFF !important;
    border: none !important;
    outline: none !important;
    border-radius: 16px !important;
    width: 46px !important;
    height: 46px !important;
    min-width: 46px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    opacity: 1 !important;
    cursor: pointer !important;
    transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1), background-color 0.2s ease !important;
    box-shadow: none !important;
}

[data-testid="stChatInputSubmitButton"]:hover {
    transform: scale(1.06);
    background-color: #333333 !important;
    border: none !important;
}

[data-testid="stChatInputSubmitButton"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
    width: 20px !important;
    height: 20px !important;
}

@keyframes bubblePopIn {
    0% { opacity: 0; transform: translateY(22px) scale(0.95); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes chatFadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes faceFloat {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-3px) rotate(2deg); }
}

@keyframes typingBounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40% { transform: translateY(-6px); opacity: 1; }
}

@keyframes pulseDot {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.35); opacity: 0.55; }
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2),
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
        position: static !important;
    }
    .hero-stack {
        height: auto;
        min-height: 480px;
    }
    .chunky-bubble {
        width: 94%;
        min-height: 115px;
    }
    .user-bubble, .assistant-bubble, [data-testid="stExpander"] {
        max-width: 95%;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

FACES = {
    "yellow": '<svg class="cute-face" width="42" height="30" viewBox="0 0 44 32" fill="none"><rect x="10" y="3" width="7" height="11" rx="3.5" fill="#111111"/><rect x="27" y="3" width="7" height="11" rx="3.5" fill="#111111"/><ellipse cx="22" cy="22" rx="7.5" ry="6" fill="#111111"/><ellipse cx="22" cy="22" rx="3" ry="2.2" fill="#FFA800"/></svg>',
    "orange": '<svg class="cute-face" width="42" height="28" viewBox="0 0 44 30" fill="none"><rect x="11" y="4" width="6.5" height="10" rx="3.2" fill="#111111"/><rect x="26.5" y="4" width="6.5" height="10" rx="3.2" fill="#111111"/><path d="M16 19C17.5 23 26.5 23 28 19" stroke="#111111" stroke-width="4.2" stroke-linecap="round"/></svg>',
    "blue": '<svg class="cute-face" width="44" height="30" viewBox="0 0 46 32" fill="none"><circle cx="14" cy="12" r="4" fill="#111111"/><ellipse cx="30" cy="9" rx="4.2" ry="5.5" fill="#111111"/><path d="M17 24C20 20 26 20 29 23" stroke="#111111" stroke-width="4.2" stroke-linecap="round"/></svg>',
    "cyan": '<svg class="cute-face" width="44" height="30" viewBox="0 0 46 32" fill="none"><path d="M10 8L17 12" stroke="#111111" stroke-width="4" stroke-linecap="round"/><path d="M36 8L29 12" stroke="#111111" stroke-width="4" stroke-linecap="round"/><circle cx="23" cy="21" r="6.5" fill="#111111"/><circle cx="23" cy="21" r="2.4" fill="#38C6F4"/></svg>',
}


def _format_text(text: str) -> str:
    """사용자/AI 텍스트를 안전하게 HTML 줄바꿈·볼드 처리합니다."""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return escaped.replace("\n", "<br>")


def render_chat_bubble(role: str, content: str) -> None:
    """실제 질문(파란색)과 답변(화이트)을 깔끔한 말풍선으로 렌더링합니다."""
    if role == "user":
        bubble_cls = "user-bubble"
        role_label = "질문"
    else:
        bubble_cls = "assistant-bubble"
        role_label = "RFP 어시스턴트"

    bubble_html = _clean_html(
        f"""
        <div class="chat-bubble {bubble_cls}">
        <div class="chat-meta">{role_label}</div>
        <div class="chat-body">{_format_text(content)}</div>
        </div>
        """
    )
    st.markdown(bubble_html, unsafe_allow_html=True)


def render_empty_hero() -> None:
    """대화 시작 전 4단 캐릭터 말풍선 프리뷰를 보여줍니다."""
    hero_html = _clean_html(
        f"""
        <div class="hero-stack">
        <div class="bubble-anim-wrap" style="animation-delay: 0.05s;">
        <div class="chunky-bubble theme-yellow zigzag-left hero-bubble">
        <div class="face-bottom">{FACES["orange"]}</div>
        <div class="hero-hint">나라장터 제안요청서(RFP) 100건을 학습한 입찰 어시스턴트입니다</div>
        </div>
        </div>
        <div class="bubble-anim-wrap" style="animation-delay: 0.14s;">
        <div class="chunky-bubble theme-orange zigzag-right hero-bubble">
        <div class="face-top">{FACES["yellow"]}</div>
        <div class="hero-hint">사업 예산, 기간, 핵심 요구사항, 참가 자격을 물어보세요!</div>
        </div>
        </div>
        <div class="bubble-anim-wrap" style="animation-delay: 0.23s;">
        <div class="chunky-bubble theme-blue zigzag-left hero-bubble">
        <div class="face-bottom">{FACES["orange"]}</div>
        <div class="hero-hint">원문 근거 청크와 출처 문서 파일(.hwp / .pdf)까지 정확히 찾아드립니다</div>
        </div>
        </div>
        <div class="bubble-anim-wrap" style="animation-delay: 0.32s;">
        <div class="chunky-bubble theme-cyan zigzag-right hero-bubble">
        <div class="face-top">{FACES["yellow"]}</div>
        <div class="hero-hint">오른쪽 추천 질문을 누르거나 아래 입력창에 질문을 입력해 보세요</div>
        </div>
        </div>
        </div>
        """
    )
    st.markdown(hero_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 상태 관리 (세션별 thread_id 유지 및 초기화)
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    new_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "faq_sample" not in st.session_state:
    st.session_state.faq_sample = random.sample(FAQ_POOL, k=min(8, len(FAQ_POOL)))

left_col, right_col = st.columns([2.1, 1.05], gap="large")

selected_example = None

with right_col:
    hdr_col, refresh_col = st.columns([1, 1], gap="small")
    with hdr_col:
        st.markdown(
            '<div class="faq-box-title">💡 가장 많이 하는 질문</div>',
            unsafe_allow_html=True,
        )
    with refresh_col:
        if st.button("↻", key="shuffle_faq", help="다른 질문 8개 보기"):
            st.session_state.faq_sample = random.sample(FAQ_POOL, k=min(8, len(FAQ_POOL)))
            st.rerun()

    for idx, q in enumerate(st.session_state.faq_sample):
        if st.button(q, key=f"faq_{idx}", use_container_width=True):
            selected_example = q

    _, clear_center_col, _ = st.columns([1, 2, 1])
    with clear_center_col:
        if st.button("🗑️ 대화 비우기", key="clear_chat"):
            new_chat()
            st.rerun()

with left_col:
    is_online, active_model = _check_backend_health()
    dot_cls = "online" if is_online else "offline"
    if is_online:
        badge_text = f"RAG ({html.escape(active_model)})" if active_model else "RAG"
    else:
        badge_text = "RAG (오프라인)"

    st.markdown(
        _clean_html(
            f"""
            <div class="conv-header">
            <h1 class="conv-title">#입찰메이트</h1>
            <span class="conv-badge"><span class="conv-dot {dot_cls}"></span>{badge_text}</span>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        render_empty_hero()
    else:
        for msg in st.session_state.messages:
            render_chat_bubble(msg["role"], msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                sources_list = msg["sources"]
                with st.expander(f"검색된 원문 근거 보기 ({len(sources_list)}개)", expanded=False):
                    for item in sources_list:
                        source_name = str(item.get("doc_id") or item.get("title") or "출처 미상")
                        citation = [source_name]
                        if item.get("chunk_id"):
                            citation.append(f"청크: {item['chunk_id']}")
                        page = item.get("page_num")
                        if type(page) is int and page > 0:
                            prefix = "PDF " if source_name.lower().endswith(".pdf") else ""
                            citation.append(f"{prefix}{page}쪽")
                        st.caption(" | ".join(citation))
                        st.text(str(item.get("snippet") or ""))

user_input = st.chat_input("공고에 대해 무엇이든 물어보세요...")
incoming_question = selected_example or user_input

if incoming_question and not st.session_state.get("pending_question"):
    st.session_state.messages.append({"role": "user", "content": incoming_question})
    st.session_state["pending_question"] = incoming_question
    st.session_state["scroll_to_bottom"] = True
    st.rerun()

if st.session_state.pop("scroll_to_bottom", False):
    _scroll_to_bottom()

pending_q = st.session_state.get("pending_question")
if pending_q:
    with left_col:
        loading_slot = st.empty()
        loading_slot.markdown(
            _clean_html(
                """
                <div class="chat-bubble assistant-bubble">
                <div class="chat-meta">RFP 어시스턴트 · 검색 중</div>
                <div class="typing-dots"><span></span><span></span><span></span></div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        _scroll_to_bottom()

        try:
            response = httpx.post(
                API_URL,
                json={"question": pending_q, "thread_id": st.session_state.thread_id},
                timeout=90.0,
            )
            if response.status_code >= 400:
                detail = ""
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = response.text
                raise RuntimeError(f"서버 오류({response.status_code}): {detail}")

            data = response.json()
            answer = data.get("answer", "답변을 생성하지 못했습니다.")
            sources = data.get("sources", [])

            st.session_state.pop("pending_question", None)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.session_state["scroll_to_bottom"] = True
            loading_slot.empty()
            st.rerun()

        except httpx.ConnectError:
            st.session_state.pop("pending_question", None)
            loading_slot.empty()
            st.error(
                "⚠️ 백엔드 서버(`http://localhost:8000`)에 연결할 수 없습니다. "
                "터미널에서 `uv run uvicorn rfp_agent.api.main:app --reload`를 먼저 실행해 주세요."
            )
        except Exception as exc:
            st.session_state.pop("pending_question", None)
            loading_slot.empty()
            st.error(f"⚠️ {exc}")