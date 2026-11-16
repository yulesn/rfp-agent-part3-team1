"""Read the selected invocation and show review needs without declaring truth.

No model, key, network, source modification or directory-wide "newest run"
selection is used. The live folder belongs to the current notebook namespace.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .review import assess_review


def load_current_review(live_folder, offline_folder):
    """Reload a result and its exact bound review after an out-of-order rerun."""
    folder=Path(live_folder if live_folder is not None else offline_folder)
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    review_path=folder/'human_review.json'
    review=json.loads(review_path.read_text(encoding='utf-8'))
    return {'folder':folder,'report':report,'review_path':review_path,
            'review':review,'assessment':assess_review(report,review)}


def comparison_checklist(report):
    """Expose the comparison's source distinctions for explicit human review.

    Exact source quotes are evidence candidates, not automatic entailment or
    completeness judgements. No case IDs or expected facts affect this view.
    """
    rows=[]
    for row in report.get('rows',[]):
        if row.get('feature')!='comparison' or row.get('skipped_reason'):continue
        result=row.get('result') or {}
        generation=result.get('generation') or {}
        parts=generation.get('document_requests') or [generation]
        adjustments=[copy.deepcopy(p['response_status_adjustment']) for p in parts
                     if p.get('response_status_adjustment')]
        duration=[]
        for claim in result.get('claims',[]):
            if claim.get('category')!='duration':continue
            duration.append({'doc_id':claim.get('doc_id'),'text':claim.get('text'),
                'citations':[{'quote':c.get('quote'),'locator':c.get('locator'),
                    'source_ranges':copy.deepcopy(c.get('source_ranges',[])),
                    'source_context':copy.deepcopy(c.get('source_context'))}
                    for c in claim.get('citations',[])]})
        rows.append({'row_id':row['row_id'],'status':result.get('status'),
            'missing_information':copy.deepcopy(result.get('missing_information',[])),
            'duration_claims':duration,
            'comparison_audit':copy.deepcopy(result.get('comparison_audit')),
            'conservative_status_adjustments':adjustments,
            'document_missing_scope_audits':[{'doc_ids':copy.deepcopy(d.get('doc_ids',[])),
                'missing_scope_audit':copy.deepcopy(d.get('missing_scope_audit'))}
                for d in result.get('document_validation',[]) if d.get('missing_scope_audit')],
            'review_checks':[
                '미확인 항목이 다른 비교 문서의 정보를 잘못 요구하지 않는지 확인',
                '본문 계약기간과 별첨 추정기간·구성기간·요구 산출물을 구분했는지 확인',
                '기산점·안정화·조정 조건을 원문과 대조하고 임의 환산이 없는지 확인'],
            'automatic_semantic_pass':False})
    return rows


def final_verification(report,review):
    """Summarize execution separately from the exact result's human verdict."""
    assessed=assess_review(report,review)
    rows=[r for r in report.get('rows',[]) if not r.get('skipped_reason')]
    results=[r.get('result') or {} for r in rows]
    bad={'error','validation_failed','incomplete','not_approved'}
    errors=sum(r.get('execution_status') in bad for r in results)
    expected=report.get('planned_qa_turns_per_profile')
    live=report.get('enabled') is True
    actual_live=live and report.get('execution_origin')=='live_openai'
    complete=(actual_live and len(report.get('profiles',[]))==1 and len(rows)==expected and not report.get('halted_at_row')
        and not report.get('coverage',{}).get('skipped_rows')
        and errors==0 and all(r.get('scope_match') is True for r in rows)
        and all(r.get('execution_status') in {'completed','not_requested'} for r in results))
    state=('execution_complete' if complete else 'execution_needs_attention') if live else 'offline_only'
    if live and not actual_live:state='test_execution_only'
    comparisons=comparison_checklist(report)
    return {'evaluation_status':report.get('status'),
        'execution_origin':report.get('execution_origin'),'execution_status':state,
        'question_turns':len(rows),'blocked_or_failed_turns':errors,
        'api_calls':report.get('usage',{}).get('api_calls',0),
        'comparison_rows_with_missing_information':sum(bool(r['missing_information']) for r in comparisons),
        'human_review_status':assessed['status'],'reviewed_rows':assessed['reviewed_rows'],
        'total_reviewable_rows':assessed['total_reviewable_rows'],
        'whole_sample_accuracy':assessed['whole_sample_accuracy'],
        'automatic_semantic_pass':False,
        'scope':'This selected development invocation only; execution success is not answer correctness.'}
