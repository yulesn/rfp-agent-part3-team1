"""[프론트·백엔드·평가] 평가 스크립트

eval/questions.jsonl 의 질문을 에이전트에 넣고, evaluation/metrics.py 의 지표로 채점해서
eval/results/ 에 결과(질문별 답변·점수 + 평균 점수)를 저장합니다.

실행:
  uv run python scripts/evaluate.py --config configs/default.yaml
  uv run python scripts/evaluate.py --config configs/default.yaml --limit 3   # 3문항 빠른 테스트
  uv run python scripts/evaluate.py --config configs/default.yaml --llm-judge # LLM 채점 포함
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from rfp_agent.agent.graph import build_graph
from rfp_agent.config import load_config
from rfp_agent.evaluation.metrics import (
    estimate_token_cost,
    hit_at_k,
    llm_judge,
    mrr,
    recall_at_k,
    retrieval_scores,
    score_answer,
)
from rfp_agent.llm import get_llm


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def _load_valid_doc_ids(documents_path: str = "data/processed/documents.jsonl") -> set[str]:
    path = Path(documents_path)
    if not path.exists():
        return set()
    doc_ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                doc_ids.add(json.loads(line)["doc_id"])
    return doc_ids


def _extract_tool_info(
    messages: list[Any], valid_doc_ids: set[str]
) -> tuple[list[str], list[tuple[str, list[str]]], str, int, int]:
    """메시지 기록에서 검색된 doc_id 목록(전체·도구 호출별), 근거 텍스트, 총 입·출력 토큰 수를 추출합니다."""
    found: list[str] = []
    calls: list[tuple[str, list[str]]] = []  # 도구 호출 순서대로 (도구 이름, 그 호출이 돌려준 doc_id 순위)
    contexts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    def _add(doc_id: str | None) -> None:
        if doc_id and doc_id not in found:
            found.append(doc_id)

    for msg in messages:
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None) or {}
            prompt_tokens += int(usage.get("input_tokens", 0))
            completion_tokens += int(usage.get("output_tokens", 0))

        if isinstance(msg, ToolMessage):
            raw = _content_to_str(msg.content)
            contexts.append(raw[:1500])

            # search_rfp 는 검색된 doc_id 목록을 artifact 로 돌려줍니다 (순위 그대로).
            artifact = getattr(msg, "artifact", None)
            if isinstance(artifact, dict):
                call_ids = [d for d in artifact.get("doc_ids", []) if d]
                calls.append((msg.name or "", list(dict.fromkeys(call_ids))))
                for doc_id in call_ids:
                    _add(doc_id)

            try:
                parsed = json.loads(raw)
                items = parsed if isinstance(parsed, list) else parsed.get("results", [parsed])
                for item in items:
                    if isinstance(item, dict):
                        _add(item.get("doc_id") or item.get("source") or item.get("metadata", {}).get("doc_id"))
            except Exception:
                pass

            for doc_id in valid_doc_ids:
                if doc_id in raw:
                    _add(doc_id)

    return found, calls, "\n---\n".join(contexts)[:3000], prompt_tokens, completion_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="RFP RAG 에이전트 자동 평가 스크립트")
    parser.add_argument("--config", default="configs/default.yaml", help="설정 파일 경로")
    parser.add_argument("--questions", default="eval/questions.jsonl", help="평가셋 JSONL 경로")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N문항만 실행 (0이면 전체)")
    parser.add_argument("--llm-judge", action="store_true", help="LLM-as-a-judge 채점 활성화")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = str(config.get("llm", {}).get("model", "gpt-4o-mini"))
    valid_doc_ids = _load_valid_doc_ids()
    top_k = int(config.get("retrieval", {}).get("top_k", 5))

    questions_path = Path(args.questions)
    with questions_path.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    if args.limit > 0:
        questions = questions[: args.limit]

    print(f"평가 시작 | 설정: {args.config} ({model_name}) | 문항 수: {len(questions)}개")
    graph = build_graph(config)

    # 채점 모델은 설정 파일의 judge_llm 섹션으로만 정합니다 (provider 도 직접 적기).
    judge_llm = None
    if args.llm_judge:
        judge_section = config.get("judge_llm")
        if not judge_section:
            raise SystemExit(f"--llm-judge 를 쓰려면 {args.config} 에 judge_llm 섹션(provider, model)을 넣어 주세요.")
        judge_llm = get_llm({**config, "llm": dict(judge_section)})

    rows: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, item in enumerate(questions, start=1):
        q_id = item["id"]
        q_type = item.get("type", "single_doc")
        question = item["question"]
        answer_doc_ids = item.get("answer_doc_ids", [])

        start = time.perf_counter()
        error_msg = ""
        prediction = ""
        retrieved_ids: list[str] = []
        calls: list[tuple[str, list[str]]] = []
        context_text = ""
        in_tok = 0
        out_tok = 0

        try:
            result = graph.invoke({"messages": [("user", question)]})
            messages = result.get("messages", [])
            if messages:
                prediction = _content_to_str(messages[-1].content)
            retrieved_ids, calls, context_text, in_tok, out_tok = _extract_tool_info(messages, valid_doc_ids)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        latency = round(time.perf_counter() - start, 2)
        cost_usd = estimate_token_cost(in_tok, out_tok, model=model_name)

        # 검색 한 번의 top-k 기준 (유형별 계산은 metrics.retrieval_scores 참고)
        ret = retrieval_scores(q_type, calls, answer_doc_ids, k=top_k)
        h_score, mrr_score, r_score = ret["hit_at_k"], ret["mrr"], ret["recall_at_k"]
        # 비교용: 에이전트가 모든 도구 호출에서 본 문서 전체 기준
        h_any = hit_at_k(retrieved_ids, answer_doc_ids)
        r_any = recall_at_k(retrieved_ids, answer_doc_ids)
        ans_res = score_answer(item, prediction)
        ans_score = ans_res["answer_score"]

        judge_res: dict[str, Any] | None = None
        if judge_llm is not None and not error_msg:
            ref_text = (
                json.dumps(item["answer_facts"], ensure_ascii=False)
                if q_type == "summary"
                else str(item.get("answer", ""))
            )
            judge_res = llm_judge(question, prediction, ref_text, context=context_text, llm=judge_llm)

        row = {
            "id": q_id,
            "type": q_type,
            "question": question,
            "answer_doc_ids": " | ".join(answer_doc_ids),
            "retrieved_doc_ids": " | ".join(retrieved_ids),
            "hit_at_k": "" if h_score is None else h_score,
            "mrr": "" if mrr_score is None else mrr_score,
            "recall_at_k": "" if r_score is None else r_score,
            "hit_any": "" if h_any is None else h_any,
            "recall_any": "" if r_any is None else r_any,
            "tool_calls": " > ".join(f"{name}({len(ids)})" for name, ids in calls),
            "answer_score": ans_score,
            "judge_reason": "" if judge_res is None else judge_res.get("reason", ""),
            "judge_accuracy": "" if judge_res is None else judge_res["accuracy"],
            "judge_faithfulness": "" if judge_res is None else judge_res["faithfulness"],
            "judge_score": "" if judge_res is None else judge_res["score"],
            "latency_sec": latency,
            "tokens": in_tok + out_tok,
            "cost_usd": cost_usd,
            "detail": error_msg or ans_res["detail"],
            "prediction": prediction.replace("\n", " "),
        }
        rows.append(row)
        by_type[q_type].append(row)

        hit_disp = "-" if h_score is None else f"{h_score:.1f}"
        mrr_disp = "-" if mrr_score is None else f"{mrr_score:.2f}"
        print(
            f"[{idx:02d}/{len(questions):02d}] {q_id} ({q_type:<17}) | "
            f"Hit: {hit_disp} | MRR: {mrr_disp} | Ans: {ans_score:.2f} | {latency:.2f}s"
            + (f" | ERR: {error_msg}" if error_msg else "")
        )

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(args.config).stem
    csv_path = results_dir / f"{timestamp}_{config_name}.csv"
    summary_path = results_dir / f"{timestamp}_{config_name}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hit_vals = [r["hit_at_k"] for r in rows if r["hit_at_k"] != ""]
    mrr_vals = [r["mrr"] for r in rows if r["mrr"] != ""]
    rec_vals = [r["recall_at_k"] for r in rows if r["recall_at_k"] != ""]
    hit_any_vals = [r["hit_any"] for r in rows if r["hit_any"] != ""]
    ans_vals = [r["answer_score"] for r in rows]
    lat_vals = [r["latency_sec"] for r in rows]
    judge_vals = [r["judge_score"] for r in rows if r["judge_score"] != ""]
    total_cost = round(sum(r["cost_usd"] for r in rows), 6)

    summary = {
        "config": args.config,
        "model": model_name,
        "timestamp": timestamp,
        "total_questions": len(rows),
        "avg_hit_at_k": round(sum(hit_vals) / len(hit_vals), 4) if hit_vals else 0.0,
        "avg_mrr": round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0.0,
        "avg_recall_at_k": round(sum(rec_vals) / len(rec_vals), 4) if rec_vals else 0.0,
        "avg_hit_any": round(sum(hit_any_vals) / len(hit_any_vals), 4) if hit_any_vals else 0.0,
        "avg_answer_score": round(sum(ans_vals) / len(ans_vals), 4) if ans_vals else 0.0,
        "avg_llm_judge": round(sum(judge_vals) / len(judge_vals), 4) if judge_vals else None,
        "avg_latency_sec": round(sum(lat_vals) / len(lat_vals), 2) if lat_vals else 0.0,
        "total_cost_usd": total_cost,
        "by_type": {
            t: {
                "count": len(items),
                "avg_answer_score": round(sum(x["answer_score"] for x in items) / len(items), 4),
                "avg_latency_sec": round(sum(x["latency_sec"] for x in items) / len(items), 2),
            }
            for t, items in sorted(by_type.items())
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print(f"평가 완료 리포트 ({config_name} / {model_name})")
    print(f"- 총 문항 수       : {summary['total_questions']}개")
    print(f"- 검색 Hit@K       : {summary['avg_hit_at_k']:.2%}")
    print(f"- 검색 MRR         : {summary['avg_mrr']:.4f}")
    print(f"- 검색 Recall@K    : {summary['avg_recall_at_k']:.2%}")
    print(f"- (참고) 본 문서 전체 Hit : {summary['avg_hit_any']:.2%}")
    print(f"- 평균 답변 점수   : {summary['avg_answer_score']:.2%}")
    if summary["avg_llm_judge"] is not None:
        print(f"- LLM Judge 점수   : {summary['avg_llm_judge']:.2%}")
    print(f"- 평균 응답 시간   : {summary['avg_latency_sec']:.2f}초 (총 비용: ${total_cost:.4f})")
    print("-" * 68)
    for t, stat in summary["by_type"].items():
        print(f"  · {t:<18} ({stat['count']:2d}문항) : 답변 {stat['avg_answer_score']:.2%} | 평균 {stat['avg_latency_sec']:.2f}초")
    print("=" * 68)
    print(f"상세 CSV 저장됨 : {csv_path}")
    print(f"요약 JSON 저장됨: {summary_path}")


if __name__ == "__main__":
    main()
