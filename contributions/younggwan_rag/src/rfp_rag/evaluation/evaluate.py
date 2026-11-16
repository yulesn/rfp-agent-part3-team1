"""Measure the deployed Phase6 route, sharing one bounded generation client."""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time
import uuid

from rfp_rag.generation.costs import estimate_usage_cost
from rfp_rag.conversation.runtime import GroundedBackend,make_conversation
from rfp_rag.conversation.session import ConversationSession
from . import VERSION
from .evidence import digest,facts_for_step,latency,micro_recall,prepare_alignment,user_steps,score_retrieval
from .review import review_template


COUNTERS=('api_calls','live_api_calls','mock_calls','cache_hits','failed_attempts',
    'attempted_input_bytes','reported_input_tokens','reported_output_tokens',
    'reported_total_tokens','reported_cached_input_tokens','usage_unknown_attempts','request_latency_ms')
PROFILES=('literal_bm25','phase6_rrf')


class LiteralBackend(GroundedBackend):
    """Ablation: same Phase6 routing, literal current query, BM25, no expansion."""
    def answer_plan(self,route):
        result=self(route['standalone_query'],doc_ids=route['effective_doc_ids'],
            feature=route['feature'],search_queries=[route['query']])
        result.setdefault('retrieval',{})['content_query_plan']={
            'queries':[route['query']],'topics':[],'version':'literal-bm25-ablation'}
        return result


def _citation_audit(env,result):
    claims=result.get('claims') or []
    if not claims:
        return {'declared_citations':0,'validated_citations':0,'valid':None,
                'semantic_correctness':None,
                'withheld_after_validation_failure':result.get('execution_status')=='validation_failed'}
    scope=result.get('effective_doc_ids') or result.get('active_doc_ids') or [result.get('doc_id')]
    try:
        core=[{'doc_id':c['doc_id'],'category':c['category'],'text':c['text'],
            'citations':[{'chunk_id':q['chunk_id'],'quote':q['quote']} for q in c.get('citations',[])]} for c in claims]
        checked=env['registry'].validate_answer({'status':result['status'],'claims':core,
            'missing_information':result.get('missing_information',[])},
            result.get('source_chunks',[]),allowed_doc_ids=scope)
        valid=checked['valid'] is True
        count=sum(len(c.get('citations',[])) for c in core)
        return {'declared_citations':count,'validated_citations':count if valid else 0,
                'valid':valid,'semantic_correctness':None}
    except (ValueError,TypeError,KeyError):
        return {'declared_citations':sum(len(c.get('citations',[])) for c in claims),
                'validated_citations':0,'valid':False,'semantic_correctness':None}


def _code_binding(root):
    package=Path(__file__).resolve().parents[1]
    files=[p for p in package.rglob('*.py') if not p.name.startswith('test_')]
    return digest({p.relative_to(package).as_posix():digest(p.read_text(encoding='utf-8')) for p in sorted(files)})


def _load_summary(env,folder,did):
    """Read and freshly validate the existing latest Phase5 content revision."""
    if folder is None:return None,'summary_folder_not_supplied'
    folder=Path(folder)
    if not (folder/'content_documents'/did/'latest.json').is_file():
        return None,'no_saved_content_revision'
    from rfp_rag.summary_content.storage import load_saved_result
    result,_=load_saved_result(folder,did,content=True)
    if env['registry'].source_hashes.get(did)!=result.get('source_sha256'):
        raise ValueError('summary_source_version_mismatch')
    chunks=env['index'].get_document_chunks(did)
    checked=env['registry'].validate_chunks(chunks,allowed_doc_ids=[did])
    if set(result.get('source_chunk_ids',[]))!={c['chunk_id'] for c in checked}:
        raise ValueError('summary_document_chunks_changed')
    result=copy.deepcopy(result)
    result['source_chunks']=checked
    return result,None


def _expected(case,tid):
    if case['feature']=='followup':
        t=next(t for t in case['turns'] if t['turn_id']==tid)
        return t.get('expected_status'),t.get('expected_doc_ids')
    status=case.get('expected_status')
    return status,[] if status=='needs_clarification' else case.get('target_doc_ids')


def _row(case,tid,profile,result,elapsed,delta,alignment,env):
    status,scope=_expected(case,tid)
    generated=bool(result.get('answer_generated'))
    reviewable=(generated or result.get('status') in {'needs_clarification','insufficient_evidence'})
    # Disabled retrieval is evidence measurement, never an answer-quality result.
    if result.get('execution_status')=='disabled':reviewable=False
    if result.get('execution_status') in {'error','validation_failed','incomplete','not_approved'}:reviewable=False
    audit=_citation_audit(env,result)
    formats=sorted({env['registry'].documents_by_id[did].get('detected_format','unknown')
        for did in env['registry'].resolve_doc_ids(scope or result.get('effective_doc_ids',[]))})
    return {'row_id':f"{profile}/{case['case_id']}/{tid}",'case_id':case['case_id'],
        'turn_id':tid,'profile':profile,'feature':case['feature'],
        'question':next((t['user'] for t in case.get('turns',[]) if t['turn_id']==tid),case['question']),
        'expected_status':status,'expected_scope':scope,
        'document_formats':formats,
        'expected_facts':copy.deepcopy(facts_for_step(case,tid)),
        'required_behaviors':copy.deepcopy(case.get('required_behaviors',[])),
        'status_match':result.get('status')==status if result.get('execution_status')!='disabled' else None,
        'status_match_is_answer_accuracy':False,
        'scope_match':set(result.get('effective_doc_ids',[]))==set(env['registry'].resolve_doc_ids(scope)) if scope is not None else None,
        'latency_ms':elapsed,'usage_delta':delta,'citation_audit':audit,
        'retrieval_score':score_retrieval(case,tid,result.get('source_chunks',[]),alignment),
        'reviewable':reviewable,'answer_correctness':None,'result':result}


def summarize(report):
    measured=[r for r in report['rows'] if not r.get('skipped_reason')]
    out={}
    for profile in report['profiles']:
        rows=[r for r in measured if r['profile']==profile and r['feature']!='summary']
        citations=[r['citation_audit'] for r in rows if r.get('citation_audit')]
        needed=sum(c['declared_citations'] for c in citations)
        valid=sum(c['validated_citations'] for c in citations)
        scopes=[r['scope_match'] for r in rows if r.get('scope_match') is not None]
        statuses=[r['status_match'] for r in rows if r.get('status_match') is not None]
        live=[r['latency_ms'] for r in rows if r['usage_delta'].get('live_api_calls',0)>0
              and not r['usage_delta'].get('cache_hits',0)]
        cache=[r['latency_ms'] for r in rows if r['usage_delta'].get('cache_hits',0)>0
               and not r['usage_delta'].get('live_api_calls',0)]
        mixed=[r['latency_ms'] for r in rows if r['usage_delta'].get('live_api_calls',0)>0
               and r['usage_delta'].get('cache_hits',0)>0]
        out[profile]={'turn_count':len(rows),'source_recovery':micro_recall(rows),
            'recall_at_k':{str(k):micro_recall(rows,k) for k in (3,5,8)},
            'source_overlap_mrr':sum(r['retrieval_score']['reciprocal_rank_first_source_overlap'] for r in rows
                if r['retrieval_score']['reciprocal_rank_first_source_overlap'] is not None)/len([r for r in rows
                if r['retrieval_score']['reciprocal_rank_first_source_overlap'] is not None]) if any(
                r['retrieval_score']['reciprocal_rank_first_source_overlap'] is not None for r in rows) else None,
            'scope_matches':sum(scopes),'scope_evaluable_turns':len(scopes),
            'status_matches':sum(statuses),'status_evaluable_turns':len(statuses),
            'citation_source_validity':valid/needed if needed else None,
            'declared_citations':needed,'invalid_citation_rows':sum(c['valid'] is False for c in citations),
            'validation_failed_turns':sum(r['result'].get('execution_status')=='validation_failed' for r in rows),
            'generation_error_turns':sum(r['result'].get('execution_status')=='error' for r in rows),
            'citation_validity_scope':'Accepted citations only; withheld validation failures counted separately.',
            'end_to_end_latency':latency([r['latency_ms'] for r in rows]),
            'fresh_live_latency':latency(live),'local_cache_latency':latency(cache),
            'mixed_live_cache_latency':latency(mixed),
            'by_feature':{feature:micro_recall([r for r in rows if r['feature']==feature])
                          for feature in sorted({r['feature'] for r in rows})},
            'by_format':{fmt:micro_recall([r for r in rows if '+'.join(r['document_formats'])==fmt])
                         for fmt in sorted({'+'.join(r['document_formats']) for r in rows if r['document_formats']})}}
    report['metrics']=out
    report['coverage']={'selected_cases':len(report['case_ids']),
        'rows':len(report['rows']),'measured_rows':len(measured),
        'skipped_rows':dict(Counter(r['skipped_reason'] for r in report['rows'] if r.get('skipped_reason'))),
        'imported_summary_rows':sum(r['feature']=='summary' for r in measured),
        'reviewable_rows':sum(r.get('reviewable',False) for r in measured)}
    report['semantic_accuracy']=None
    report['status']='live_review_pending' if report['enabled'] else 'offline_measured'
    return report


def run_evaluation(env,cases,*,profiles=('literal_bm25','phase6_rrf'),enabled=False,
                   case_ids=None,approved_doc_ids=None,max_calls=8,max_input_bytes=600000,
                   summary_folder=None,pricing=None,client=None,progress=None):
    """No gold target IDs, facts or expected answers enter the production path.

    One generation client is retained across all cases. A fresh conversation per
    case isolates user history without resetting the client's usage budget.
    """
    cases=list(cases)
    if type(enabled)is not bool:raise TypeError('enabled는 bool이어야 합니다.')
    if any(c.get('split')!='dev' for c in cases):raise ValueError('dev 전용 평가입니다.')
    if not profiles or len(set(profiles))!=len(profiles) or set(profiles)-set(PROFILES):raise ValueError('평가 프로필이 잘못됐습니다.')
    if case_ids is not None:
        if not case_ids or len(set(case_ids))!=len(case_ids):raise ValueError('평가 사례를 중복 없이 선택하세요.')
        by_id={c['case_id']:c for c in cases}
        if set(case_ids)-set(by_id):raise ValueError('없는 평가 사례 ID입니다.')
        cases=[by_id[k] for k in case_ids]
    if enabled and (case_ids is None or list(profiles)!=['phase6_rrf']):
        raise ValueError('실제 평가는 case_ids를 지정하고 phase6_rrf 프로필만 실행하세요.')
    if enabled and any(c['feature']=='summary' for c in cases):
        raise ValueError('전체 요약은 저장된 Phase5 결과를 가져옵니다. API 재생성하지 않습니다.')
    if enabled and any((c.get('context') or {}).get('tool_results') for c in cases):
        raise ValueError('합성 파싱 장애 사례는 이 실제 검색 평가의 실행 대상이 아닙니다.')
    if enabled and sum(len(user_steps(c)['steps']) for c in cases)>max_calls:
        raise ValueError('선택한 질문 턴 수가 호출 상한을 넘습니다. 사례 수를 줄이세요.')
    if not cases:raise ValueError('평가할 사례가 없습니다.')
    for case in cases:
        inputs=user_steps(case)
        if inputs['selected_doc_ids']:env['registry'].resolve_doc_ids(inputs['selected_doc_ids'])
    if pricing is not None:
        # Validate configured arithmetic without assuming that rates are current.
        estimate_usage_cost({},pricing)
        model=env.get('phase5_config',{}).get('generation',{}).get('model')
        if pricing.get('model')!=model:raise ValueError('단가 모델과 생성 모델이 다릅니다.')
    settings=json.loads((Path(env['root'])/'configs/phase6.json').read_text(encoding='utf-8'))
    # Routing needs identity/title metadata, not another copy of every source
    # block and reference view for each of the many independent dev cases.
    # The original registry/index still supplies and validates all evidence.
    routing_keys={'doc_id','canonical_doc_id','sha256','title','file_name',
                  'agency','document_year','detected_format','is_duplicate_alias'}
    routing_documents=[{k:copy.deepcopy(v) for k,v in d.items() if k in routing_keys} for d in env['documents']]
    generation={**env.get('phase5_config',{}).get('generation',{}),
                'max_calls':max_calls,'max_total_input_bytes':max_input_bytes}
    base=make_conversation({**env,'documents':routing_documents},enabled=enabled,approved_doc_ids=approved_doc_ids,
        retrieval=settings['retrieval'],config=settings['conversation'],generation=generation,client=client)
    alignment=prepare_alignment(env,cases)
    descriptor={'version':VERSION,'case_ids':[c['case_id'] for c in cases],
        'suite_fingerprint':digest(cases),'index_fingerprint':env['index'].get_metadata()['index_fingerprint'],
        'routing_catalog_fingerprint':base.catalog_fingerprint,
        'code_fingerprint':_code_binding(env['root']),'profiles':list(profiles),
        'settings':settings,'generation':generation,'enabled':enabled,'pricing':pricing,
        'execution_origin':'injected_test_client' if client is not None else 'live_openai' if enabled else 'offline',
        'evaluation_split':'dev','holdout_question_file_opened':False,
        'planned_qa_turns_per_profile':sum(len(user_steps(c)['steps']) for c in cases
            if c['feature']!='summary' and not (c.get('context') or {}).get('tool_results')),
        'baseline_protocol':'Same conversation router; literal current user query + BM25 ablation. Not a historical benchmark.',
        'python':platform.python_version(),'platform':{'system':platform.system(),'machine':platform.machine()},
        'dependencies':env['index'].get_metadata().get('dependencies',{})}
    report={**descriptor,'rows':[],'gold_used_as_generation_input':False,
        'alignment':{k:alignment.get(k) for k in ('requested_dev_links','missing_dev_alignment_links','mapped_links','chunk_source_issue_count')}}
    for profile in profiles:
        backend=base.backend if profile=='phase6_rrf' else LiteralBackend(env,base.llm,
            retrieval={**settings['retrieval'],'method':'bm25'})
        for case in cases:
            cid=case['case_id'];context=case.get('context') or {}
            if progress:progress(profile,cid)
            if context.get('tool_results'):
                report['rows'].append({'row_id':f'{profile}/{cid}/fixture','profile':profile,'case_id':cid,
                    'feature':case['feature'],'skipped_reason':'synthetic_parser_fixture_not_simulated','reviewable':False})
                continue
            inputs=user_steps(case)
            if case['feature']=='summary':
                if profile!='phase6_rrf':continue
                scope=env['registry'].resolve_doc_ids(inputs['selected_doc_ids'])
                result=None;reason='summary_requires_one_user_selected_document'
                try:
                    if len(scope)==1:result,reason=_load_summary(env,summary_folder,scope[0])
                except (ValueError,TypeError,KeyError,OSError):reason='saved_summary_validation_failed'
                if result is None:
                    report['rows'].append({'row_id':f'{profile}/{cid}/summary','profile':profile,'case_id':cid,
                        'feature':'summary','skipped_reason':reason,'reviewable':False})
                else:
                    row=_row(case,'t1',profile,result,0.0,dict.fromkeys(COUNTERS,0),alignment,env)
                    row['historical_generation_usage_not_rebilled']=True
                    row['summary_field_states']={f['field_id']:f['state'] for f in result.get('fields',[])}
                    report['rows'].append(row)
                continue
            session=ConversationSession(routing_documents,backend=backend,
                config=base.config)
            if inputs['filters']:session.set_filters(inputs['filters'])
            if inputs['selected_doc_ids']:session.select_documents(inputs['selected_doc_ids'])
            for step in inputs['steps']:
                before=base.llm.usage_report();started=time.perf_counter()
                result=session.ask(step['user'])
                elapsed=(time.perf_counter()-started)*1000;after=base.llm.usage_report()
                delta={k:after.get(k,0)-before.get(k,0) for k in COUNTERS}
                row=_row(case,step['turn_id'],profile,result,elapsed,delta,alignment,env)
                if context.get('candidate_doc_ids'):row['candidate_fixture_note']='Production catalog used; candidate list not injected as active selection.'
                report['rows'].append(row)
                # Stop provider work after the first service/budget failure;
                # no automatic retry and no fallback model is used.
                if enabled and result.get('execution_status')=='error':
                    report['halted_at_row']=row['row_id'];break
            if report.get('halted_at_row'):break
        if report.get('halted_at_row'):break
    report['usage']=base.llm.usage_report()
    report['cost_estimate']=estimate_usage_cost(report['usage'],pricing) if pricing else None
    report['pricing_status']='configured_rates_not_verified_current' if pricing else 'not_configured_token_usage_reported'
    report['run_fingerprint']=digest({'descriptor':descriptor,'results':[{'row_id':r['row_id'],
        'result':r.get('result'),'skipped_reason':r.get('skipped_reason')} for r in report['rows']]})
    return summarize(report)


def save_report(root,report):
    """Append a local run, never overwrite a prior result or human judgement."""
    folder=Path(root)/'phase7_outputs/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8])
    payloads={'report.json':report,'human_review.json':review_template(report)}
    encoded={name:json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n' for name,value in payloads.items()}
    folder.mkdir(parents=True,exist_ok=False)
    for name,text in encoded.items():(folder/name).write_text(text,encoding='utf-8')
    return folder
