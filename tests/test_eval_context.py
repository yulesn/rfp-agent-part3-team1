import importlib.util
from pathlib import Path

from langchain_core.messages import ToolMessage

_spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).parents[1] / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def _search_msg(chunks):
    return ToolMessage(
        content="본문",
        name="search_rfp",
        tool_call_id="x",
        artifact={
            "doc_ids": ["d1"] * len(chunks),
            "evidence": [{"source": "a.hwp", "chunk_id": cid, "text": text} for cid, text in chunks],
        },
    )


def test_judge_context_dedupes_repeated_chunks():
    msgs = [_search_msg([("c1", "가"), ("c2", "나")]), _search_msg([("c2", "나"), ("c3", "다")])]
    _, calls, context = evaluate._extract_tool_info(msgs, set())
    assert len(calls) == 2
    assert context.count("청크: c2") == 1
    assert "청크: c3" in context


def test_judge_context_keeps_non_search_tool_output():
    msg = ToolMessage(content="요약 결과", name="summarize_rfp", tool_call_id="y", artifact={"doc_ids": ["d1"]})
    _, _, context = evaluate._extract_tool_info([msg, msg], set())
    assert context == "요약 결과"
