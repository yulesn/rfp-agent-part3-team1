"""Score source recovery after retrieval; never use labels as search inputs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from rfp_rag.retrieval.evaluation import audit_chunk_coverage, _coverage


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_dev_cases(path):
    path=Path(path)
    if 'holdout' in path.name.casefold():
        raise ValueError('기존 holdout 파일은 Phase7 개발 평가에서 열지 않습니다.')
    rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    if not rows or any(r.get('split')!='dev' for r in rows):
        raise ValueError('명시적인 dev split만 평가할 수 있습니다.')
    ids=[r.get('case_id') for r in rows]
    if any(not isinstance(x,str) or not x or '/' in x or '\\' in x for x in ids) or len(ids)!=len(set(ids)):
        raise ValueError('평가 사례 ID가 잘못됐거나 중복됐습니다.')
    for r in rows:
        if r.get('feature') not in {'fact','comparison','followup','uncertainty','summary'}:
            raise ValueError('지원하지 않는 평가 종류입니다.')
        if not isinstance(r.get('question'),str) or not r['question'].strip():
            raise ValueError('평가 질문이 누락됐습니다.')
        if not isinstance(r.get('context',{}),dict):raise ValueError('사용자 문서 맥락이 잘못됐습니다.')
        if r['feature']=='followup':
            turns=r.get('turns')
            if not isinstance(turns,list) or not turns or any(not isinstance(t.get('user'),str) or not t['user'].strip() for t in turns):
                raise ValueError('후속 질문에는 원래 사용자 질문 턴이 필요합니다.')
            tids=[t.get('turn_id') for t in turns]
            if any(not isinstance(t,str) or not t or '/' in t for t in tids) or len(tids)!=len(set(tids)):
                raise ValueError('후속 질문 턴 ID가 잘못됐습니다.')
    return rows


def user_steps(case):
    """Return user input only, excluding targets, gold values and status labels."""
    context=case.get('context') or {}
    scope=context.get('selected_doc_ids') or []
    if not isinstance(scope,list) or any(not isinstance(x,str) for x in scope):
        raise ValueError('선택 문서 ID는 문자열 목록이어야 합니다.')
    turns=case['turns'] if case['feature']=='followup' else [{'turn_id':'t1','user':case['question']}]
    return {'selected_doc_ids':list(scope),'filters':copy.deepcopy(context.get('filters',{})),
            'steps':[{'turn_id':t['turn_id'],'user':t['user']} for t in turns]}


def facts_for_step(case,turn_id):
    facts=case.get('required_facts',[])
    if case['feature']=='followup':
        turn=next(t for t in case['turns'] if t['turn_id']==turn_id)
        chosen=set(turn.get('required_fact_ids',[]))
        return [f for f in facts if f.get('fact_id') in chosen]
    return facts


def prepare_alignment(env,cases):
    """Recheck only requested dev links against the current immutable chunks.

    The Phase2 alignment metadata is filtered by exact dev case/fact/quote/hash
    before it enters the mapper. No holdout question file is opened.
    """
    expected={}
    for case in cases:
        if case.get('split')!='dev':raise ValueError('dev 전용 원문 정렬입니다.')
        for f in case.get('required_facts',[]):
            for i,e in enumerate(f.get('evidence',[]),1):
                expected[(case['case_id'],f['fact_id'],i)]=e
    metadata=json.loads((Path(env['root'])/'data/source_evidence_validation.json').read_text(encoding='utf-8'))
    records=[];seen=set()
    for r in metadata.get('records',[]):
        key=(r.get('case_id'),r.get('fact_id'),r.get('evidence_index'))
        e=expected.get(key)
        if r.get('evidence_kind')!='fact' or not e or key in seen:continue
        if (r.get('doc_id')!=e.get('doc_id') or r.get('quote')!=e.get('quote')
            or r.get('gold_sha256')!=e.get('sha256')):continue
        records.append(r);seen.add(key)
    audit=audit_chunk_coverage(env['documents'],env['chunks'],{'records':records})
    if audit['chunk_source_issue_count']:
        raise ValueError('현재 청크와 원문 위치 검사에 실패했습니다.')
    audit['missing_dev_alignment_links']=len(expected)-len(seen)
    audit['requested_dev_links']=len(expected)
    return audit


def score_retrieval(case,turn_id,chunks,alignment,cutoffs=(3,5,8)):
    """Recall requires complete gold quotation intervals, including conditions."""
    labels={r['link_id']:r for r in alignment['records']}
    links=[]
    for f in facts_for_step(case,turn_id):
        for i,_ in enumerate(f.get('evidence',[]),1):
            links.append(labels.get(f"{case['case_id']}|fact|{f['fact_id']}|{i}"))
    mapped=sum(bool(l and l.get('mapped')) for l in links)

    def measure(selected):
        ids={c['chunk_id'] for c in selected}
        measured=[_coverage(l,ids) for l in links if l and l.get('mapped')]
        recovered=sum(ok for _,ok in measured)
        return {'expected_links':len(links),'mapped_links':mapped,
            'unmapped_links':len(links)-mapped,'recovered_links':recovered,
            'source_span_recall':recovered/len(links) if links and mapped==len(links) else None,
            'mean_source_character_coverage':statistics.mean(v for v,_ in measured) if measured and mapped==len(links) else None}

    first=next((1/i for i,c in enumerate(chunks,1)
        if any(_coverage(l,{c['chunk_id']})[0]>0 for l in links if l and l.get('mapped'))),0.0)
    return {'packed_context':measure(chunks),'at_k':{str(k):measure(chunks[:k]) for k in cutoffs},
            'reciprocal_rank_first_source_overlap':first if links and mapped==len(links) else None,
            'context_chunk_count':len(chunks),
            'scope':'Exact gold source-character recovery in the actual packed model input; not answer correctness.'}


def latency(values):
    values=sorted(values)
    return {'count':len(values),'p50_ms':statistics.median(values) if values else None,
        'p95_ms':values[max(0,math.ceil(.95*len(values))-1)] if values else None,
        'total_ms':sum(values),'sampling':'One pass of ask(), prepared conversation and open index; initialization excluded.'}


def micro_recall(rows,cutoff=None):
    parts=[r['retrieval_score']['packed_context'] if cutoff is None else r['retrieval_score']['at_k'][str(cutoff)]
           for r in rows if r.get('retrieval_score')]
    needed=sum(r['expected_links'] for r in parts)
    mapped=sum(r['mapped_links'] for r in parts)
    recovered=sum(r['recovered_links'] for r in parts)
    return {'expected_links':needed,'mapped_links':mapped,'unmapped_links':needed-mapped,
            'recovered_links':recovered,'micro_source_span_recall':recovered/needed if needed and mapped==needed else None}
