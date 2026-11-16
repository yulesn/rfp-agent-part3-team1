"""질문별 근거가 검색되는지 확인합니다. 실제 임베딩 API를 호출합니다."""

import argparse
import json
from pathlib import Path

from rfp_agent.config import PROJECT_ROOT, load_config
from rfp_agent.retrieval.retriever import retrieve


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_questions(path: Path) -> list[dict]:
    questions = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not questions:
        raise ValueError("질문 파일이 비어 있습니다.")
    for row in questions:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
        ):
            raise ValueError("question 문자열이 필요합니다.")
        for name in ("answer_doc_ids", "evidence"):
            values = row.get(name)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(x, str) or not x.strip() for x in values)
            ):
                raise ValueError(f"{name}에는 비어 있지 않은 문자열 목록이 필요합니다.")
        if "max_rank" in row and (type(row["max_rank"]) is not int or row["max_rank"] < 1):
            raise ValueError("max_rank는 1 이상의 정수여야 합니다.")
    return questions


def check_questions(questions: list[dict], config: dict) -> dict:
    results = []
    for question in questions:
        documents = retrieve(question["question"], config)
        # 파일명뿐 아니라 같은 청크 안에 기대한 근거 문장이 있는지 확인합니다.
        matches = [
            {"rank": rank, "doc_id": doc.metadata["doc_id"], "chunk_id": doc.metadata["chunk_id"]}
            for rank, doc in enumerate(documents, 1)
            if rank <= question.get("max_rank", config["retrieval"]["top_k"])
            and doc.metadata["doc_id"] in question["answer_doc_ids"]
            and all(text in doc.page_content for text in question["evidence"])
        ]
        results.append(
            {
                "id": question.get("id"),
                "question": question["question"],
                "passed": bool(matches),
                "matches": matches,
                "retrieved_doc_ids": [doc.metadata["doc_id"] for doc in documents],
            }
        )
    return {
        "passed": all(row["passed"] for row in results),
        "top_k": config["retrieval"]["top_k"],
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrieval_demo.yaml")
    parser.add_argument("--questions", default="examples/retrieval_demo/questions.jsonl")
    parser.add_argument("--report")
    args = parser.parse_args()
    try:
        questions = read_questions(project_path(args.questions))
        report = check_questions(questions, load_config(args.config))
        if args.report:
            output = project_path(args.report)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 -- CLI 경계에서 외부 오류 본문을 숨깁니다.
        # 키가 포함될 수 있는 외부 오류 응답은 출력하지 않습니다.
        parser.exit(
            1, f"검색 확인 실패: {type(exc).__name__}, HTTP {getattr(exc, 'status_code', None)}\n"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
