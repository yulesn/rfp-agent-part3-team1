"""[LLM·에이전트] 터미널에서 에이전트에게 질문하기

UI 없이 에이전트를 빠르게 확인할 때 씁니다. 먼저 build_index.py 로 벡터DB 를 만들어 두어야 합니다.

실행:  uv run python scripts/ask.py "OO시청 공고의 사업 예산은 얼마야?"
대화:  uv run python scripts/ask.py --chat          (후속 질문 이어서 하기, 빈 줄이나 exit 로 종료)
"""

import argparse

from rfp_agent.agent.graph import build_graph, new_memory
from rfp_agent.config import load_config


def chat(config: dict) -> None:
    graph = build_graph(config, checkpointer=new_memory())
    thread = {"configurable": {"thread_id": "cli"}}
    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in ("", "exit", "quit"):
            break
        result = graph.invoke({"messages": [("user", question)]}, thread)
        print(result["messages"][-1].content)
        if result.get("current_doc_id"):
            print(f"\n(대화 중인 공고: {result['current_doc_id']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--chat", action="store_true", help="후속 질문을 이어서 하는 대화 모드")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.chat:
        chat(config)
        return
    if not args.question:
        parser.error("질문을 입력하거나 --chat 으로 실행하세요.")

    graph = build_graph(config)
    result = graph.invoke({"messages": [("user", args.question)]})

    # 중간 과정(도구 호출)까지 보고 싶으면 result["messages"] 전체를 출력해 보세요.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
