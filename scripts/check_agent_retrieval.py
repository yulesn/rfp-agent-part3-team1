"""팀 LangGraph의 본문 검색 → 답변 연결 확인. --live를 지정할 때만 API를 호출합니다."""

import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from retrieval_index import inspect_index

from rfp_agent.agent.graph import build_graph
from rfp_agent.agent.tools.search_rfp import search_rfp
from rfp_agent.config import PROJECT_ROOT, load_config


def describe_run(messages: list) -> dict:
    """도구 호출·응답·최종 답변을 기록합니다. 답변의 의미적 정답 여부는 판정하지 않습니다."""
    calls = [call for msg in messages if isinstance(msg, AIMessage) for call in msg.tool_calls]
    responses = [msg for msg in messages if isinstance(msg, ToolMessage)]
    completed = {msg.tool_call_id for msg in responses if msg.status == "success"}
    last = messages[-1] if messages else None
    answer = last.content if isinstance(last, AIMessage) and not last.tool_calls else ""
    usage = {key: 0 for key in ("input_tokens", "output_tokens", "total_tokens")}
    llm_calls = 0
    usage_complete = True
    for msg in messages:
        if isinstance(msg, AIMessage):
            llm_calls += 1
            usage_complete = usage_complete and msg.usage_metadata is not None
            for key in usage:
                usage[key] += (msg.usage_metadata or {}).get(key, 0)
    return {
        "connection_passed": bool(calls and answer) and all(
            call["name"] == "search_rfp" and call["id"] in completed for call in calls
        ),
        "tool_calls": calls,
        "tool_responses": [
            {"name": msg.name, "status": msg.status, "content": msg.content} for msg in responses
        ],
        "answer": answer,
        "llm_successful_responses": llm_calls,
        "llm_usage": usage if usage_complete else None,
        "embedding_usage_included": False,
        "semantic_accuracy": None,
        "scope": "Team LangGraph with search_rfp only; not the standalone younggwan_rag integration.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question", default="한영대학교 트랙운영 학사정보시스템 고도화의 사업 예산과 기간은?")
    parser.add_argument("--live", action="store_true", help="질문 임베딩과 실제 LLM 답변 생성 실행")
    parser.add_argument("--report", help="실행 결과 JSON 경로. API 키는 저장하지 않습니다.")
    args = parser.parse_args()
    try:
        path = None
        if args.report:
            path = Path(args.report)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                parser.error("보고서가 이미 있습니다. 새 --report 경로를 지정하세요.")
        if not args.question.strip():
            parser.error("질문을 입력하세요.")
        config = load_config(args.config)
        state = inspect_index(config)
        if state["state"] != "ready":
            parser.error("완성된 DB가 필요합니다. DB 공유본과 설정 경로를 확인하세요.")
        report = {
            "config": args.config,
            "index": state,
            "api_called": False,
            "question": args.question,
        }
        if args.live:
            if "llm" not in config:
                parser.error("llm 설정이 필요합니다. configs/default.yaml을 사용하세요.")
            # 미완성 조건 검색·요약 도구는 이 연결 시험에 등록하지 않습니다.
            graph = build_graph(config, tools=[search_rfp])
            result = graph.invoke(
                {"messages": [("user", args.question)]}, {"recursion_limit": 8}
            )
            report.update(describe_run(result["messages"]))
            report.update(
                api_called=True,
                llm={key: config["llm"][key] for key in ("provider", "model")},
                embedding={key: config["embedding"][key] for key in ("provider", "model")},
            )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 실수로 기존 키·입력·보고서를 덮어쓰지 않습니다.
            with path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        summary = {key: value for key, value in report.items() if key != "tool_responses"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.live and not report["connection_passed"]:
            parser.exit(1, "연결 확인 실패: 검색 도구 호출·응답·최종 답변을 확인하세요.\n")
        if not args.live:
            print("API 미호출. 실제 검색·답변까지 확인하려면 --live를 추가하세요.")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"실행 실패: {type(exc).__name__}. 설정과 출력 경로를 확인하세요.\n")
    except Exception as exc:  # noqa: BLE001 -- 외부 응답·키 일부가 오류에 포함될 수 있음
        parser.exit(
            1, f"실행 실패: {type(exc).__name__}, HTTP {getattr(exc, 'status_code', None)}\n"
        )


if __name__ == "__main__":
    main()
