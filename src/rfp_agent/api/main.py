"""[프론트·백엔드·평가] FastAPI 서버

- 에이전트(build_graph)를 HTTP API로 감싸는 파일입니다. UI와 평가 스크립트는 이 API를 호출합니다.
- 그래프와 벡터DB는 서버가 켜질 때 한 번만 만들고, 요청마다 재사용합니다.

실행:  uv run uvicorn rfp_agent.api.main:app --reload --port 8000
확인:  브라우저에서 http://localhost:8000/docs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.messages import ToolMessage

from rfp_agent.agent.graph import build_graph
from rfp_agent.api.schemas import AskRequest, AskResponse, Source
from rfp_agent.config import load_config

app = FastAPI(title="RFP RAG 에이전트 API")

_graph = None
_doc_titles: dict[str, str] | None = None


def _load_doc_titles() -> dict[str, str]:
    global _doc_titles
    if _doc_titles is not None:
        return _doc_titles

    titles: dict[str, str] = {}
    for candidate in (Path("data/processed/documents.jsonl"), Path("data/documents.jsonl")):
        if candidate.exists():
            with candidate.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    doc_id = row.get("doc_id", "")
                    meta = row.get("metadata") or {}
                    titles[doc_id] = meta.get("title") or doc_id.rsplit(".", 1)[0]
            break
    _doc_titles = titles
    return _doc_titles


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(load_config())
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
    doc_titles = _load_doc_titles()
    sources: list[Source] = []
    seen: set[tuple[str, str]] = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        raw = _content_to_str(msg.content)

        # JSON 형태 도구 결과 파싱
        try:
            parsed = json.loads(raw)
            items = parsed if isinstance(parsed, list) else parsed.get("results", [parsed])
            for item in items:
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata") or {}
                doc_id = item.get("doc_id") or item.get("source") or meta.get("doc_id") or ""
                snippet = str(item.get("snippet") or item.get("text") or item.get("content") or "")[:300]
                title = item.get("title") or meta.get("title") or doc_titles.get(doc_id, doc_id)
                if doc_id and (doc_id, snippet[:60]) not in seen:
                    seen.add((doc_id, snippet[:60]))
                    sources.append(Source(doc_id=doc_id, title=title, snippet=snippet))
            if sources:
                continue
        except Exception:
            pass

        # 일반 텍스트 도구 결과에서 doc_id 매칭
        for doc_id, title in doc_titles.items():
            if doc_id in raw and (doc_id, raw[:60]) not in seen:
                seen.add((doc_id, raw[:60]))
                sources.append(Source(doc_id=doc_id, title=title, snippet=raw[:300]))

    return sources


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
    try:
        graph = get_graph()
        result = graph.invoke({"messages": [("user", req.question)]})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"에이전트 실행 오류: {exc}") from exc

    messages = result.get("messages", [])
    answer = _content_to_str(messages[-1].content) if messages else ""
    sources = _extract_sources(messages)

    return AskResponse(answer=answer, sources=sources)