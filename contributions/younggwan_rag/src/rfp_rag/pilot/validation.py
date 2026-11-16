"""A bounded, repeatable pilot exercise; automatic checks are not answer scores.

Cases use caller-selected document context only. No gold answers, target IDs,
source quotes or prior assistant text enter routing or generation. Human
review stays bound to each exact result. This is a diagnostic exercise, not
an unseen-test benchmark.
"""
from __future__ import annotations

import copy
from pathlib import Path

from rfp_rag.evaluation.evaluate import _citation_audit
from .pilot import PilotSession, digest, load_json, read_record, session_report, write_new
from .claim_audit import claim_citation_audit, claim_review_summary


def validation_plan(doc_ids):
    """Only document IDs explicitly selected by the caller are used as context."""
    if not isinstance(doc_ids, list) or not doc_ids or len(doc_ids) > 2 or len(set(doc_ids)) != len(doc_ids):
        raise ValueError('검증할 대표 문서를 중복 없이 한 개 또는 두 개 선택하세요.')
    a = [doc_ids[0]]
    cases = [
        {'id': 'completion', 'select': a, 'question': '선택한 사업의 완료·검수 조건이 뭐야?',
         'topic': 'acceptance', 'review_hint': '검수 선행 조건과 완료 후 보고·지원 의무를 구분한다.'},
        {'id': 'scope_followup', 'question': '그러니까, 구체적인 공사의 내용이 무엇이냐는 거지.',
         'topic': 'work_scope', 'forbidden_topic': 'acceptance',
         'review_hint': '구체적인 대상 업무·기능·사업 범위를 답한다. 일반 개발 방향만으로 대신하지 않는다. 건설공사인지 정보시스템 사업인지는 원문으로 확인한다.'},
        {'id': 'budget', 'question': '사업예산은 얼마이고 부가세가 포함되나요?', 'topic': 'budget',
         'review_hint': '금액·단위·VAT 조건을 원문 그대로 유지한다.'},
        {'id': 'duration', 'question': '그건 몇 개월짜리야? 안정화가 포함되는지도 알려줘.', 'topic': 'duration',
         'forbidden_topic': 'budget', 'review_hint': '기산점·기간·안정화 포함 여부·예외를 구분한다. 근거에 없는 포함 여부는 유보한다.'},
        {'id': 'award', 'question': '최종 낙찰업체와 실제 계약금액을 확정해줘.',
         'review_hint': '실제 낙찰·계약 결과 근거가 없으면 예산으로 대체하지 않고 유보한다. RFP 전체에 결과가 없다고 단정하지 않는다.'}]
    if len(doc_ids) == 2:
        cases += [
            {'id': 'switch', 'select': [doc_ids[1]], 'question': '그 사업의 예산과 부가세 조건은?',
             'topic': 'budget', 'review_hint': '앞 문서의 금액·인용·사업 내용을 새 문서에 가져오지 않는다.'},
            {'id': 'comparison', 'select': list(doc_ids),
             'question': '두 사업의 예산과 사업기간을 조건을 구분해 비교해줘.',
             'topic': 'budget', 'review_hint': '각 문서의 금액·기간·기산점·조건을 분리하고, 본문 기간과 별첨 산정기간을 혼합하지 않는다.'}]
    cases.append({'id': 'no_selection', 'reset': True, 'question': '그 사업의 예산은?',
        'expected_status': 'needs_clarification', 'must_not_generate': True,
        'review_hint': '문서를 선택하지 않았으므로 먼저 대상 문서를 확인한다.'})
    return {'version': 'phase8-validation-v1', 'selected_doc_ids': list(doc_ids), 'cases': cases,
        'maximum_generation_requests': 8 if len(doc_ids) == 2 else 5,
        'scope': 'Caller-selected pilot diagnostics; novelty and semantic accuracy are unverified.'}


def run_validation(env, doc_ids, *, enabled=False, approved_doc_ids=None, client=None):
    if type(enabled) is not bool:
        raise TypeError('enabled는 True/False여야 합니다.')
    selected = env['registry'].resolve_doc_ids(doc_ids)
    plan = validation_plan(selected)
    approved = env['registry'].resolve_doc_ids(approved_doc_ids) if approved_doc_ids else []
    if enabled and not set(selected).issubset(approved):
        raise ValueError('실제 검증에 사용할 모든 문서를 API 허용 목록에 명시하세요.')
    pilot = PilotSession(env, client=client)
    # This exercise owns one client and one budget across all cases. Never
    # increase the configured maximum to make a plan fit.
    if enabled and pilot.chat.llm.config.max_calls < plan['maximum_generation_requests']:
        raise ValueError('전체 검증의 최대 호출 수가 설정된 세션 상한을 넘습니다.')
    pilot.configure_generation(enabled=enabled, approved_doc_ids=approved)
    plan['plan_fingerprint'] = digest(plan)
    write_new(pilot.folder / 'validation_plan.json', plan)
    rows = []
    current_scope = []
    for case in plan['cases']:
        if case.get('reset'):
            pilot.reset_conversation()
            current_scope = []
        if 'select' in case:
            pilot.select(case['select'], filters={})
            current_scope = list(case['select'])
        record = pilot.submit(case['question'], request_id='validation:' + case['id'])
        result = record['result']
        topics = (result.get('retrieval', {}).get('content_query_plan') or {}).get('topics', [])
        checks = {'document_scope': set(result.get('effective_doc_ids', [])) == set(current_scope),
            'source_scope': all(c['doc_id'] in current_scope for c in result.get('source_chunks', [])),
            'execution_without_error': result.get('execution_status') not in {'error', 'validation_failed', 'incomplete', 'not_approved'}}
        if case.get('topic'):
            checks['requested_topic'] = case['topic'] in topics
        if case.get('forbidden_topic'):
            checks['previous_topic_not_inherited'] = case['forbidden_topic'] not in topics
        if case.get('expected_status'):
            checks['routing_status'] = result.get('status') == case['expected_status']
        if case.get('must_not_generate'):
            checks['no_generation'] = record['usage_delta']['api_calls'] == 0 and not result.get('answer_generated')
        audit = _citation_audit(env, result)
        if audit['valid'] is not None:
            checks['accepted_citations_valid'] = audit['valid'] is True
        rows.append({'case_id': case['id'], 'turn_id': record['turn_id'],
            'result_fingerprint': record['fingerprint'], 'checks': checks,
            'citation_audit': audit, 'review_hint': case['review_hint'],
            'claim_citation_audit': claim_citation_audit(result),
            'answer_correctness': None, 'semantic_correctness_verified': False})
        if pilot.halted:
            break
    report = {'version': 'phase8-validation-v1', 'plan_fingerprint': plan['plan_fingerprint'],
        'session_id': pilot.session_id, 'enabled': enabled, 'execution_origin': pilot.chat.llm.usage_report()['execution_mode'],
        'planned_cases': len(plan['cases']), 'executed_cases': len(rows), 'rows': rows,
        'mechanical_checks_passed': len(rows) == len(plan['cases']) and all(all(r['checks'].values()) for r in rows),
        'semantic_accuracy': None, 'independent_quality_verified': False,
        'human_review_status': 'review_pending', 'automatic_semantic_pass': False,
        'scope': plan['scope']}
    report['report_fingerprint'] = digest(report)
    write_new(pilot.folder / 'validation_report.json', report)
    return pilot.folder, validation_status(pilot.folder)


def validation_status(folder, *, include_reviews=True):
    """Refresh explicit human reviews while checking the immutable result links."""
    folder = Path(folder)
    report = load_json(folder / 'validation_report.json')
    if report.get('report_fingerprint') != digest({k: v for k, v in report.items() if k != 'report_fingerprint'}):
        raise ValueError('검증 보고서의 지문이 다릅니다.')
    plan = load_json(folder / 'validation_plan.json')
    payload = {k: v for k, v in plan.items() if k != 'plan_fingerprint'}
    if (plan.get('plan_fingerprint') != digest(payload)
            or report.get('plan_fingerprint') != plan['plan_fingerprint']):
        raise ValueError('다른 검증 계획 또는 변경된 계획입니다.')
    diagnostics = []
    for row in report['rows']:
        record = read_record(folder / 'turns' / row['turn_id'])
        if record['fingerprint'] != row['result_fingerprint'] or record['session_id'] != report['session_id']:
            raise ValueError('검증 결과와 실제 답변의 연결이 다릅니다.')
        answer = record['result']
        validation = answer.get('validation') or {}
        # Quotations in citation_diagnostics are rejected model data. Expose
        # indices/outcomes only; never present those quotations as an answer.
        citation_checks = [{k: copy.deepcopy(item[k]) for k in (
            'claim_index', 'citation_index', 'outcome', 'reason', 'chunk_id',
            'claim_doc_id', 'fragment_count', 'reference_rebuilt') if k in item}
            for item in validation.get('citation_diagnostics', [])]
        diagnostics.append({'case_id': row['case_id'], 'turn_id': row['turn_id'],
            'status': answer.get('status'), 'execution_status': answer.get('execution_status'),
            'validation_valid': validation.get('valid'),
            'validation_errors': copy.deepcopy(validation.get('errors', [])),
            'citation_recovery': copy.deepcopy(validation.get('citation_recovery', {})),
            'citation_checks': citation_checks,
            'document_validation': [{k: copy.deepcopy(item[k]) for k in
                ('doc_ids', 'valid', 'errors', 'citation_recovery') if k in item}
                for item in answer.get('document_validation', [])],
            'error_diagnostic': copy.deepcopy(answer.get('error_diagnostic')),
            'model_request_fingerprint': (answer.get('generation') or {}).get('request_fingerprint'),
            'claim_citation_audit': claim_citation_audit(answer),
            'rejected_model_text_displayed': False})
    result = copy.deepcopy(report)
    result['claim_citation_review'] = claim_review_summary([d['claim_citation_audit'] for d in diagnostics])
    if include_reviews:
        result['session'] = session_report(folder)
        result['human_review_status'] = result['session']['human_review_status']
    result['diagnostics'] = {'version': 'phase8-saved-diagnostics-v1',
        'session_id': report['session_id'], 'diagnostic_api_calls': 0, 'read_only': True,
        'planned_cases': report['planned_cases'], 'executed_cases': report['executed_cases'],
        'mechanical_checks_passed': report['mechanical_checks_passed'], 'turns': diagnostics,
        'semantic_correctness_verified': False}
    return result
