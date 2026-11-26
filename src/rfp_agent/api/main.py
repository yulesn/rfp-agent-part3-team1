"""[프론트·백엔드·평가] FastAPI 서버

실행:  uv run uvicorn rfp_agent.api.main:app --reload --port 8000
확인:  브라우저에서 http://localhost:8000/docs
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, ToolMessage

from rfp_agent.agent.graph import build_graph, new_memory
from rfp_agent.api.schemas import AskRequest, AskResponse, Source
from rfp_agent.config import load_config

app = FastAPI(title="RFP RAG 에이전트 API")

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(load_config(), checkpointer=new_memory())
    return _graph


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b["text"]) if isinstance(b, dict) and "text" in b else str(b)
            for b in content
        )
    return str(content or "")


def _extract_sources(messages: list[Any]) -> list[Source]:
    """이번 질문(마지막 HumanMessage 이후) search_rfp artifact['evidence']만 추출합니다."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            break
    else:
        return []

    unique: dict[tuple[Any, Any], Source] = {}
    for message in messages[index + 1 :]:
        if not isinstance(message, ToolMessage) or message.name != "search_rfp":
            continue
        if isinstance(message.artifact, dict):
            for item in message.artifact.get("evidence", []):
                if not isinstance(item, dict):
                    continue
                source = str(item.get("source") or "").strip()
                chunk_id = item.get("chunk_id")
                page_num = item.get("page_num")
                text = str(item.get("text") or "").strip()

                key = (source, chunk_id or text)
                if (source or text) and key not in unique:
                    unique[key] = Source(
                        doc_id=source,
                        title=source,
                        snippet=text,
                        chunk_id=chunk_id,
                        page_num=page_num if type(page_num) is int else None,
                    )

    return list(unique.values())


@app.get("/health")
def health() -> dict:
    """서버가 살아 있는지와 현재 연결된 LLM 모델명을 확인합니다."""
    cfg = load_config()
    llm_cfg = cfg.get("llm") or {}
    return {
        "status": "ok",
        "provider": llm_cfg.get("provider", ""),
        "model": llm_cfg.get("model", ""),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    thread_id = req.thread_id or str(uuid4())
    try:
        graph = get_graph()
        result = graph.invoke(
            {"messages": [("user", req.question)]},
            {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 16,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"에이전트 실행 오류: {exc}") from exc

    messages = result.get("messages", [])
    answer = _content_to_str(messages[-1].content) if messages else ""
    sources = _extract_sources(messages)

    return AskResponse(answer=answer, sources=sources)