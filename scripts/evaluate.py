"""[프론트·백엔드·평가] 평가 스크립트

eval/questions.jsonl 의 질문을 에이전트에 넣고, evaluation/metrics.py 의 지표로 채점해서
eval/results/ 에 결과(질문별 답변·점수 + 평균 점수)를 저장합니다.
설정 파일만 바꿔서 여러 번 돌리면 실험끼리 비교할 수 있습니다. 결과는 노션 「실험 로그」에 기록하세요.

실행:  uv run python scripts/evaluate.py --config configs/default.yaml
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    args = parser.parse_args()  # noqa: F841  (TODO 를 채우면서 사용)

    # TODO:
    # 1. 평가셋 읽기 (jsonl, 한 줄에 질문 하나)
    # 2. build_graph(config) 로 에이전트를 만들고 질문마다 실행 (답변, 걸린 시간 기록)
    # 3. metrics.py 의 지표로 채점
    # 4. eval/results/{날짜}_{설정이름}.csv 로 저장하고 평균 점수 출력
    raise NotImplementedError


if __name__ == "__main__":
    main()
