"""Grounded QA, document comparison and user-only follow-up state."""
from __future__ import annotations

import json
import re
import time
import copy

from rfp_rag.retrieval.session import SearchSession
from .citation_recovery import validate_with_quote_recovery
from .request_policy import award_evidence_gate, generation_question
from .comparison import document_question,local_missing_information,DurationSources
from .duration_values import missing_duration_values
from .llm import COMPARISON_DOCUMENT_TASK,safe_llm_diagnostic


DEFAULT_RETRIEVAL = {'method': 'bm25', 'top_k': 8, 'per_doc_top_k': 6,
                     'fetch_k': 40, 'mmr_lambda': .75,
                     'max_context_bytes': 42000, 'max_question_chars': 4000}


def _config(value):
    config = dict(DEFAULT_RETRIEVAL)
    supplied = dict(value or {})
    unknown = set(supplied) - set(config)
    if unknown:
        raise ValueError('Unknown answer retrieval configuration: ' + ', '.join(sorted(unknown)))
    config.update(supplied)
    for name in ['top_k', 'per_doc_top_k', 'fetch_k', 'max_context_bytes', 'max_question_chars']:
        v = config[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(name + ' must be a positive integer')
    if config['method'] not in {'bm25', 'char_tfidf', 'rrf', 'mmr'}:
        raise ValueError('Unknown retrieval method')
    if isinstance(config['mmr_lambda'], bool) or not isinstance(config['mmr_lambda'], (int, float)) or not 0 <= config['mmr_lambda'] <= 1:
        raise ValueError('mmr_lambda must be between zero and one')
    return config


def _view(chunk):
    # This is an input-size estimate, not model tokenization or pricing.
    return {key: chunk.get(key) for key in ['chunk_id', 'doc_id', 'sha256', 'file_name',
            'locator', 'source_method', 'quality_flags', 'text']}


def _size(chunk):
    return len(json.dumps(_view(chunk), ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def _pack(groups, budget):
    """Round-robin document groups before truncation; never cut source text."""
    selected, omitted, seen = [], [], set()
    used = 2
    for offset in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if offset >= len(group):
                continue
            chunk = group[offset]
            cid = chunk['chunk_id']
            if cid in seen:
                continue
            seen.add(cid)
            size = _size(chunk) + 1
            if used + size <= budget:
                selected.append(chunk)
                used += size
            else:
                omitted.append(cid)
    return selected, omitted, used


def _safe(value):
    text = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    for char in ['\\', '|', '*', '_', '`', '[', ']']:
        text = text.replace(char, '\\' + char)
    return text.replace('\n', ' / ')


def _comparison_markdown(validation, selected_doc_ids, catalog):
    """A deterministic document-column table; every factual cell keeps citations."""
    from .evidence import CATEGORIES, CATEGORY_LABELS
    by_doc = {row['doc_id']: row for row in catalog}
    docs = list(selected_doc_ids)
    claims = validation['claims']
    rows, references = [], []
    numbered = {}
    for claim in claims:
        marks = []
        for citation in claim['citations']:
            key = citation['chunk_id'], citation['quote']
            if key not in numbered:
                numbered[key] = len(references) + 1
                references.append(citation)
            marks.append(f'[{numbered[key]}]')
        role=claim.get('duration_source_role')
        category=('duration_estimate' if role=='duration_estimate' else
                  'duration_mixed' if role=='duration_mixed' else claim['category'])
        rows.append((claim['doc_id'], category, _safe(claim['text']) + ' ' + ''.join(marks)))
    lines = ['| 항목 | ' + ' | '.join(_safe(by_doc.get(doc, {}).get('file_name', doc)) for doc in docs) + ' |',
             '| --- | ' + ' | '.join('---' for _ in docs) + ' |']
    labels={**CATEGORY_LABELS,'duration_estimate':'기간 산정자료(별첨)',
            'duration_mixed':'기간 근거 혼합 — 추가 확인'}
    categories=list(CATEGORIES)
    at=categories.index('duration')+1
    categories[at:at]=['duration_estimate','duration_mixed']
    for category in categories:
        if not any(row[1] == category for row in rows):
            continue
        cells = ['; '.join(row[2] for row in rows if row[0] == doc and row[1] == category)
                 or '현재 근거에서 미확인' for doc in docs]
        lines.append('| ' + labels[category] + ' | ' + ' | '.join(cells) + ' |')
    if validation['missing_information']:
        lines.extend(['', '추가 확인이 필요한 항목:', ''])
        lines.extend('- ' + _safe(item) for item in validation['missing_information'])
    if references:
        lines.extend(['', '원문 근거:', ''])
        for number, citation in enumerate(references, 1):
            lines.append(f'{number}. ' + _safe(citation['file_name']) + ' · ' + _safe(citation['locator'])
                         + ' — “' + _safe(citation['quote']) + '”')
    lines.extend(['', '인용문과 원문 위치를 대조했습니다. 인용문이 주장 전체를 뒷받침하는지는 별도 검토가 필요합니다.'])
    return '\n'.join(lines)


def _generation_report(parts):
    """Keep child request metadata and usage; never retain model payloads."""
    if not parts:return None
    if len(parts)==1:return copy.deepcopy(parts[0])
    modes={p.get('execution_mode') for p in parts}
    usages=[p.get('usage',{}) for p in parts];usage={}
    for k in {k for u in usages for k in u}:
        values=[u.get(k) for u in usages]
        if k=='usage_reported':usage[k]=all(v is True for v in values)
        elif all(type(v) in (int,float) for v in values):usage[k]=sum(values)
        else:usage[k]=None
    return {'execution_mode':next(iter(modes)) if len(modes)==1 else 'mixed',
        'live_api':any(p.get('live_api') for p in parts),
        'cache_hit':all(p.get('cache_hit') for p in parts),'usage':usage,
        'request_fingerprints':[p.get('request_fingerprint') for p in parts],
        'document_requests':copy.deepcopy(parts),'request_count':len(parts),
        'request_latency_ms':sum(p.get('request_latency_ms',0) for p in parts)}


class AnswerSession:
    """Uses only user history, never previous assistant answers or eval labels."""

    def __init__(self, index, llm, registry, config=None, active_doc_ids=None):
        self.index, self.llm, self.registry = index, llm, registry
        self.config = _config(config)
        self.retrieval_session = SearchSession(index, active_doc_ids=active_doc_ids)

    @property
    def active_doc_ids(self):
        return self.retrieval_session.active_doc_ids

    @property
    def query_history(self):
        return self.retrieval_session.query_history

    def select_documents(self, ids):
        return self.retrieval_session.select_documents(ids)

    def _empty(self, status, execution_status, message, retrieval=None):
        return {'status': status, 'execution_status': execution_status,
                'answer_generated': False, 'claims': [], 'markdown': message,
                'missing_information': [], 'active_doc_ids': self.active_doc_ids,
                'retrieval': retrieval or {}, 'source_chunks': [], 'generation': None,
                'validation': None, 'semantic_entailment_verified': False}

    def ask(self, query, *, feature='qa', candidate_doc_ids=None, request_question=None):
        if not isinstance(query, str):
            raise TypeError('질문은 문자열이어야 합니다.')
        if feature not in {'qa', 'fact', 'comparison', 'followup', 'uncertainty'}:
            raise ValueError('전체 요약은 FullDocumentSummarizer 경로를 사용하세요.')
        if len(query) > self.config['max_question_chars']:
            raise ValueError('질문 길이가 설정 상한을 넘었습니다.')
        started = time.perf_counter()
        search = self.retrieval_session.ask(query, top_k=self.config['top_k'],
            method=self.config['method'], candidate_doc_ids=candidate_doc_ids)
        if search['status'] != 'retrieved':
            return self._empty(search['status'], 'not_requested', search.get('reason') or '문서 범위와 질문을 확인하세요.', search)
        comparison = feature == 'comparison' or bool(re.search(r'비교|차이|두\s*사업', query))
        scope = self.active_doc_ids
        if comparison and len(scope) < 2:
            return self._empty('needs_clarification', 'not_requested',
                               '비교할 문서를 두 개 이상 명시적으로 선택해 주세요.', search)
        groups = []
        if comparison:
            # Each document earns a separate quota BEFORE merged context packing.
            for doc_id in scope:
                hits = self.index.search(search['standalone_query'],
                    top_k=self.config['per_doc_top_k'], allowed_doc_ids=[doc_id],
                    method=self.config['method'], fetch_k=self.config['fetch_k'],
                    mmr_lambda=self.config['mmr_lambda'])
                groups.append(hits)
        else:
            # SearchSession manages scope/history; use configured candidate/MMR values.
            hits = self.index.search(search['standalone_query'], top_k=self.config['top_k'],
                allowed_doc_ids=scope or None, method=self.config['method'],
                fetch_k=self.config['fetch_k'], mmr_lambda=self.config['mmr_lambda'])
            groups = [hits]
        candidate_hits = [hit for group in groups for hit in group]
        # Raises before API dispatch if any hit or source provenance was altered.
        self.registry.validate_chunks(candidate_hits, allowed_doc_ids=scope or None)
        packed, omitted, input_bytes = _pack(groups, self.config['max_context_bytes'])
        context = self.registry.validate_chunks(packed, allowed_doc_ids=scope or None)
        represented = {c['doc_id'] for c in context}
        missing_docs = [doc for doc in scope if doc not in represented]
        retrieval = {**search, 'hits': candidate_hits,
            'comparison': comparison, 'method': self.config['method'],
            'context_chunk_ids': [c['chunk_id'] for c in context],
            'omitted_chunk_ids': omitted, 'context_bytes_estimate': input_bytes,
            'missing_selected_doc_ids': missing_docs,
            'document_quotas_applied_before_packing': comparison,
            'retrieval_latency_ms': (time.perf_counter() - started) * 1000}
        if not context:
            return self._empty('insufficient_evidence', 'not_requested',
                '현재 검색·입력 예산 안에서 인용할 원문을 확보하지 못했습니다.', retrieval)
        gate=award_evidence_gate(request_question or search['standalone_query'],context)
        if gate:
            result=self._empty('insufficient_evidence','not_requested',
                '현재 제공·검색된 원문 근거로는 '+', '.join(gate['requested_fields'])+
                '을 확정할 수 없습니다. RFP의 과업예산은 실제 계약금액의 근거가 아닙니다. '
                '낙찰 결과 공고 또는 체결 계약서를 추가로 확인해야 합니다.',retrieval)
            result.update(source_chunks=context,missing_information=gate['requested_fields'],
                          request_policy=gate)
            return result
        enabled = getattr(getattr(self.llm, 'config', None), 'enabled', False)
        if not enabled:
            result = self._empty('not_generated', 'disabled',
                'OpenAI 실행이 꺼져 있습니다. 원문 근거 후보를 확인한 뒤 ENABLE_OPENAI와 해당 실행 옵션을 켜세요.', retrieval)
            result['source_chunks'] = context
            return result
        catalog=[r for r in self.index.get_catalog() if r['doc_id'] in scope]
        requests=([{'task':COMPARISON_DOCUMENT_TASK,'question':generation_question(
                    document_question(search['standalone_query'],did,catalog),doc_id=did),
                    'evidence':[c for c in context if c['doc_id']==did]} for did in scope
                   if any(c['doc_id']==did for c in context)] if comparison else
                  [{'task':'qa','question':generation_question(search['standalone_query']),'evidence':context}])
        retrieval['generation_strategy']='per_document_then_validated_table' if comparison else 'scoped_qa'
        retrieval['planned_generation_requests']=len(requests)
        generated_parts=[];validations=[];document_validation=[];current_doc_ids=[]
        try:
            preflight=getattr(self.llm,'preflight_requests',None)
            if callable(preflight):preflight(requests)
            for request in requests:
                current_doc_ids=list(dict.fromkeys(c['doc_id'] for c in request['evidence']))
                generated=self.llm.generate(**request)
                generated_parts.append({k:v for k,v in generated.items() if k!='payload'})
                ids=list(dict.fromkeys(c['doc_id'] for c in request['evidence']))
                checked=validate_with_quote_recovery(self.registry,generated.get('payload'),
                    request['evidence'],allowed_doc_ids=ids)
                missing_audit=None
                if comparison and checked['valid']:
                    missing,missing_audit=local_missing_information(checked['missing_information'],ids[0],catalog)
                    checked['missing_information']=missing
                    checked['status']=('partial' if checked['claims'] and missing else 'answered'
                        if checked['claims'] else 'insufficient_evidence')
                document_validation.append({'doc_ids':ids,'valid':checked['valid'],
                    'errors':checked.get('errors',[]),
                    'missing_scope_audit':missing_audit,
                    'response_status_adjustment':generated.get('response_status_adjustment'),
                    'citation_recovery':checked.get('citation_recovery',{})})
                validations.append(checked)
                if not checked['valid']:break
        except Exception as error:
            # No upstream API body, key, or source snippet is exposed.
            diagnostic=safe_llm_diagnostic(error)
            result = self._empty('not_generated', 'error',
                '답변 생성에 실패했습니다. '+diagnostic['message'], retrieval)
            result.update({'source_chunks': context, 'error_type': type(error).__name__,
                           'error_detail':diagnostic['message'],'error_diagnostic':diagnostic,
                           'failed_request_doc_ids':current_doc_ids,
                           'document_validation':document_validation,
                           'generation':_generation_report(generated_parts),
                           'latency_ms': (time.perf_counter() - started) * 1000})
            return result
        validation=next((v for v in validations if not v['valid']),None)
        generation_report=_generation_report(generated_parts)
        if validation is None:
            if comparison:
                claims=[{'doc_id':c['doc_id'],'category':c['category'],'text':c['text'],
                         'citations':[{'chunk_id':q['chunk_id'],'quote':q['quote']} for q in c['citations']]}
                        for v in validations for c in v['claims']]
                missing=[f"문서 {d['doc_ids'][0]}: {m}" for v,d in zip(validations,document_validation)
                         for m in v['missing_information']]
                status='partial' if claims and missing else 'answered' if claims else 'insufficient_evidence'
                validation=self.registry.validate_answer({'status':status,'claims':claims,
                    'missing_information':missing},context,allowed_doc_ids=scope)
            else:validation=validations[0]
        if not validation['valid']:
            result = self._empty('not_generated', 'validation_failed',
                '생성 결과의 출처·형식 검사에 실패해 답변을 표시하지 않습니다.', retrieval)
            result.update({'validation': validation, 'generation': generation_report,
                           'document_validation':document_validation,
                           'source_chunks': context, 'latency_ms': (time.perf_counter() - started) * 1000})
            return result
        comparison_audit=[]
        if comparison:
            current_question=request_question or query
            estimate_requested=bool(re.search(r'산정|추정|적정\s*기간',current_question))
            classifiers={did:DurationSources(self.registry.documents_by_id[did]) for did in scope}
            retained=[]
            for i,claim in enumerate(validation['claims']):
                if claim['category']!='duration':retained.append(claim);continue
                audit=classifiers[claim['doc_id']].claim_role(claim)
                missing_values=missing_duration_values(claim)
                if missing_values:
                    comparison_audit.append({'claim_index':i,'doc_id':claim['doc_id'],**audit,
                        'withheld':True,'reason':'duration_value_not_in_literal_citations',
                        'unquoted_values':missing_values})
                    if audit['role']!='duration_estimate' or estimate_requested:
                        validation['missing_information'].append('기간 수치의 직접 인용 보완 필요')
                    continue
                claim['duration_source_role']=audit['role']
                comparison_audit.append({'claim_index':i,'doc_id':claim['doc_id'],**audit})
                retained.append(claim)
                if audit['role']=='duration_mixed':
                    validation['missing_information'].append('기간 본문과 산정자료가 한 주장에 혼합되어 근거 구분 확인 필요')
                    validation['status']='partial'
            validation['claims']=retained
            if (re.search(r'(?:사업|과업|계약|수행)\s*기간|몇\s*개월',current_question)
                and not re.search(r'(?:산정자료|추정기간|산정서|적정기간)\s*만',current_question)):
                for did in scope:
                    roles={c.get('duration_source_role') for c in retained
                           if c['doc_id']==did and c['category']=='duration'}
                    if roles=={'duration_estimate'}:
                        validation['missing_information'].append(f'문서 {did}: 본문의 계약·과업기간 근거 미확보 (별첨 산정자료와 구분)')
            validation['status']=('partial' if retained and validation['missing_information']
                else 'answered' if retained else 'insufficient_evidence')
        # Preserve missing evidence on either selected document; never imply a complete comparison.
        cited_docs = {claim['doc_id'] for claim in validation['claims']}
        missing_comparison = [doc for doc in scope if doc not in cited_docs] if comparison else []
        if missing_docs or missing_comparison:
            validation = dict(validation)
            validation['status'] = 'partial' if validation['claims'] else 'insufficient_evidence'
            validation['missing_information'] = list(validation['missing_information']) + [
                f'선택 문서 {doc}: 현재 입력에서 검증된 비교 근거 미확보'
                for doc in dict.fromkeys(missing_docs + missing_comparison)]
        return {'status': validation['status'], 'execution_status': 'completed',
                'answer_generated': bool(validation['claims']), 'claims': validation['claims'],
                'missing_information': validation['missing_information'],
                'markdown': (_comparison_markdown(validation, scope, self.index.get_catalog())
                             if comparison else self.registry.render(validation)), 'active_doc_ids': scope,
                'source_chunks': context, 'retrieval': retrieval, 'generation': generation_report,
                'validation': validation, 'usage': generation_report.get('usage', {}),
                'document_validation':document_validation,
                'comparison_audit':comparison_audit,
                'latency_ms': (time.perf_counter() - started) * 1000,
                'semantic_entailment_verified': False}
