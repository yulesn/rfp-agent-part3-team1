"""[LLM·에이전트] 개발용 LLM 호출 기록

- config 의 agent.show_trace 가 true 면 그래프에 이 콜백을 붙여, 질문 1번에 일어난 모든 LLM 호출
  (agent 노드 + summarize_rfp 같은 도구 안의 호출)의 입력 프롬프트·출력·시간을 그대로 기록합니다.
- nodes.py 가 최종 답변 끝에 format_calls() 결과를 붙입니다. 데모·평가 때는 show_trace: false.
"""

import json
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler


class LLMCallRecorder(BaseCallbackHandler):
    def __init__(self):
        self._lock = threading.Lock()
        self.calls: dict = {}  # run_id → 기록 (호출 순서 유지)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:  # 그래프 실행 시작 = 새 질문
            with self._lock:
                self.calls = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        meta = metadata or {}
        with self._lock:
            self.calls[run_id] = {
                "node": meta.get("langgraph_node"),
                "model": meta.get("ls_model_name"),
                "input": [(m.type, m.content) for m in messages[0]],
                "start": time.perf_counter(),
            }

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            call = self.calls.get(run_id)
            if call is None:
                return
            call["sec"] = round(time.perf_counter() - call["start"], 2)
            message = getattr(response.generations[0][0], "message", None)
            call["output"] = response.generations[0][0].text
            call["tool_calls"] = getattr(message, "tool_calls", None) or []
            usage = (getattr(message, "usage_metadata", None) or {})
            call["tokens"] = {k: usage.get(k) for k in ("input_tokens", "output_tokens")}

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            if run_id in self.calls:
                self.calls[run_id]["output"] = f"ERROR: {error!r}"

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.calls.values())


def format_calls(calls: list[dict]) -> str:
    """기록을 가공 없이 텍스트로 덤프합니다."""
    lines = [f"LLM calls: {len(calls)}"]
    for i, call in enumerate(calls, start=1):
        lines.append(f"\n===== call {i} | node={call['node']} | model={call['model']} "
                     f"| {call.get('sec', '?')}s | tokens={call.get('tokens')} =====")
        for role, content in call["input"]:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            lines.append(f"[{role}]\n{text}")
        if call.get("tool_calls"):
            lines.append("[output.tool_calls]\n" + json.dumps(
                [{"name": c["name"], "args": c["args"]} for c in call["tool_calls"]],
                ensure_ascii=False))
        lines.append(f"[output]\n{call.get('output', '(no output)')}")
    # 마크다운으로 해석되지 않도록 코드 블록에 넣습니다.
    body = "\n".join(lines).replace("```", "'''")
    return f"\n\n---\n**🔍 실행 과정 (개발용)**\n\n```text\n{body}\n```"
