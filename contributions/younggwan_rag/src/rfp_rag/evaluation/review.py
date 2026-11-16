"""An explicit human verdict bound to this exact result and reference data."""
from __future__ import annotations

import copy
from .evidence import digest


JUDGEMENTS=('answer_correct','claims_supported','conditions_preserved',
            'required_facts_complete','abstention_or_clarification_appropriate')


def review_template(report):
    rows=[]
    for r in report['rows']:
        if not r.get('reviewable'):continue
        basis={k:copy.deepcopy(r.get(k)) for k in ('row_id','question','expected_status',
            'expected_facts','required_behaviors','expected_scope','result','citation_audit')}
        rows.append({'row_id':r['row_id'],'binding':digest(basis),
            'question':r['question'],'expected_status':r['expected_status'],
            'expected_facts_for_review_only':copy.deepcopy(r.get('expected_facts',[])),
            'required_behaviors_for_review_only':copy.deepcopy(r.get('required_behaviors',[])),
            'judgements':dict.fromkeys(JUDGEMENTS),'notes':''})
    return {'version':'phase7-human-review-v1','run_fingerprint':report['run_fingerprint'],
            'reviewer':'','rows':rows,'automatic_semantic_pass':False}


def assess_review(report,review):
    expected=review_template(report)
    if (not isinstance(review,dict) or review.get('version')!=expected['version']
        or review.get('run_fingerprint')!=expected['run_fingerprint']):
        raise ValueError('다른 실행의 검토 기록입니다.')
    actual=review.get('rows')
    if not isinstance(actual,list) or len(actual)!=len(expected['rows']):raise ValueError('검토 항목 개수가 다릅니다.')
    reviewed=correct=0;failures=[];pending=[]
    for a,e in zip(actual,expected['rows']):
        if any(a.get(k)!=e.get(k) for k in e if k not in {'judgements','notes'}):
            raise ValueError('검토할 답변·원문·정답 정보의 연결이 바뀌었습니다.')
        j=a.get('judgements')
        if (not isinstance(j,dict) or set(j)!=set(JUDGEMENTS)
            or any(v is not None and type(v)is not bool for v in j.values())
            or not isinstance(a.get('notes',''),str)):
            raise ValueError('검토 판정은 true/false/null이어야 합니다.')
        if any(v is None for v in j.values()):pending.append(a['row_id']);continue
        reviewed+=1
        if all(j.values()):correct+=1
        else:failures.append({'row_id':a['row_id'],'failed':[k for k,v in j.items() if not v]})
    reviewer=review.get('reviewer')
    identified=isinstance(reviewer,str) and bool(reviewer.strip())
    total=len(actual)
    complete_sample=(report.get('enabled') is True and report.get('execution_origin')=='live_openai'
        and not report.get('halted_at_row')
        and total==report.get('planned_qa_turns_per_profile') and len(report.get('profiles',[]))==1
        and not report.get('coverage',{}).get('skipped_rows')
        and all((r.get('result',{}).get('generation') or {}).get('execution_mode')=='live_openai'
            for r in report['rows'] if r.get('result',{}).get('generation')))
    return {'status':'review_complete' if total and reviewed==total and identified else 'review_pending',
        'reviewed_rows':reviewed,'total_reviewable_rows':total,'review_coverage':reviewed/total if total else None,
        'fully_correct_rows':correct,'accuracy_on_reviewed_rows':correct/reviewed if reviewed and identified else None,
        'whole_sample_accuracy':correct/total if total and reviewed==total and identified and complete_sample else None,
        'pending_rows':pending,'failed_rows':failures,'reviewer_recorded':identified,
        'scope':'This development sample only; not unseen-test or corpus-wide accuracy.'}
