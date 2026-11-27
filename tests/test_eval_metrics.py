"""평가 지표(검색 top-k 기준, unanswerable 채점) 테스트."""

from rfp_agent.evaluation.metrics import retrieval_scores, score_unanswerable


def test_single_doc_uses_first_search_only():
    calls = [("search_rfp", ["x", "y", "z", "w", "v", "a"]), ("search_rfp", ["a"])]
    # 정답 a 는 첫 검색 6위(top-5 밖), 두 번째 검색 1위 → 첫 검색 기준이라 0
    assert retrieval_scores("single_doc", calls, ["a"], k=5) == {"hit_at_k": 0.0, "mrr": 0.0, "recall_at_k": 0.0}
    calls = [("search_rfp", ["x", "a"])]
    assert retrieval_scores("paraphrase", calls, ["a"], k=5) == {"hit_at_k": 1.0, "mrr": 0.5, "recall_at_k": 1.0}


def test_multi_doc_averages_per_answer_doc():
    calls = [("search_rfp", ["a", "x"]), ("search_rfp", ["a", "y"]), ("search_rfp", ["z"])]
    # 정답 a 는 찾음(1위), b 는 못 찾음 → hit 0, recall 0.5 (a 를 두 번 찾아도 점수 안 오름)
    assert retrieval_scores("multi_doc", calls, ["a", "b"], k=5) == {"hit_at_k": 0.0, "mrr": 0.5, "recall_at_k": 0.5}
    calls = [("search_rfp", ["x", "a"]), ("search_rfp", ["b"])]
    assert retrieval_scores("multi_doc", calls, ["a", "b"], k=5) == {"hit_at_k": 1.0, "mrr": 0.75, "recall_at_k": 1.0}


def test_summary_uses_summarize_rfp_pick():
    calls = [("search_rfp", ["x"]), ("summarize_rfp", ["a"])]
    assert retrieval_scores("summary", calls, ["a"])["hit_at_k"] == 1.0
    assert retrieval_scores("summary", [("summarize_rfp", ["x"])], ["a"])["hit_at_k"] == 0.0


def test_fallback_skips_empty_calls():
    # search_rfp 없이 filter 만 썼으면 결과가 있는 첫 호출을 봄
    calls = [("filter_rfp_by_metadata", []), ("filter_rfp_by_metadata", ["a"])]
    assert retrieval_scores("single_doc", calls, ["a"])["hit_at_k"] == 1.0
    # search_rfp 를 썼는데 비어 있으면 그대로 0 (진짜 못 찾은 것)
    calls = [("search_rfp", []), ("filter_rfp_by_metadata", ["a"])]
    assert retrieval_scores("single_doc", calls, ["a"])["hit_at_k"] == 0.0


def test_no_answer_doc_or_no_calls():
    assert retrieval_scores("unanswerable", [("search_rfp", ["a"])], [])["hit_at_k"] is None
    assert retrieval_scores("single_doc", [], ["a"])["hit_at_k"] == 0.0


def test_score_unanswerable():
    assert score_unanswerable("해외 지사 운영 예산에 관한 내용이 명시되어 있지 않습니다.") == 1.0
    assert score_unanswerable("최대 건수가 구체적인 숫자로 명시되어 있지는 않습니다.") == 1.0
    assert score_unanswerable("최종 계약금액을 확인할 수 없습니다.") == 1.0
    assert score_unanswerable("낙찰업체 정보를 찾을 수 없습니다.") == 1.0
    # 흔한 표현만 있으면 통과시키지 않음
    assert score_unanswerable("제공된 문서에 따르면 예산은 3억 원입니다. 다른 조건은 없습니다.") == 0.0
    assert score_unanswerable("") == 0.0
