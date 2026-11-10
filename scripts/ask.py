"""[LLM·에이전트] 터미널에서 에이전트에게 질문하기

UI 없이 에이전트를 빠르게 확인할 때 씁니다. 먼저 build_index.py 로 벡터DB 를 만들어 두어야 합니다.

실행:  uv run python scripts/ask.py "OO시청 공고의 사업 예산은 얼마야?"
"""

import argparse

from rfp_agent.agent.graph import build_graph
from rfp_agent.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    graph = build_graph(load_config(args.config))
    result = graph.invoke({"messages": [("user", args.question)]})

    # 중간 과정(도구 호출)까지 보고 싶으면 result["messages"] 전체를 출력해 보세요.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
