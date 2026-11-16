"""Rebuild document summaries from checked model facts and exact source text.

No provider client, reference answer or document-ID exception is used here.
Source-context rules are conservative checks, not a semantic correctness oracle.
The original generation and its accounting are immutable inputs.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import re
import unicodedata

from rfp_rag.summary_fields.fields import FIELDS, FIELD_BY_ID, parse_field_claim
from rfp_rag.summary_full.engine import _binding_citation
from rfp_rag.summary_full.review import answer_fingerprint
from . import CONTENT_VERSION


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _normal(value):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFC',value)).casefold()


_ABSENCE = re.compile(
    r'(?:제공된\s*(?:근거|원문|범위)|이\s*(?:제공\s*)?(?:묶음|범위|근거)|본\s*묶음|'
    r'구체(?:적)?\s*(?:사업명|기한|금액|목록)|사업명\s*자체|사업명\s*값).{0,160}'
    r'(?:확인되지|확인할\s*수\s*없|명시되어\s*있지|제시되지|없(?:고|음|다)|보이지)', re.S)
_LABEL_ONLY = {_normal(f['label']) for f in FIELDS} | {
    _normal(x) for x in ('사업명','사 업 명','공사','입찰참가 자격',
        '제안서 및 제출서류 안내','기술능력평가 항목 및 배점한도')}
_NOTICE = re.compile(r'(?:입찰\s*공고(?:문)?|입찰\s*유의서|공고문).{0,55}(?:참조|참고|의함|따르|따름|명기|확인)',re.S)
_FORM_DESCRIPTION = re.compile(r'서식상|서식의\s*예시|서식에.{0,40}(?:표시|기재)|'
    r'목차에.{0,35}항목|사업수행계획서\s*목차|파일명.{0,50}확인|파일명이\s*보',re.S)


def _fresh_claim(raw, document, by_id, registry, *, local=False):
    """Revalidate source identities, literal text and the exact occurrence."""
    did=document['doc_id'];sha=document['sha256']
    if raw.get('doc_id') != did:
        raise ValueError('claim_document_scope_mismatch')
    parsed=parse_field_claim(raw)
    citations=raw.get('citations')
    if not isinstance(citations,list) or not citations:
        raise ValueError('citation_missing')
    fresh=[]
    for q in citations:
        if not isinstance(q,dict) or q.get('chunk_id') not in by_id:
            raise ValueError('citation_chunk_outside_document')
        chunk=by_id[q['chunk_id']]
        if chunk.get('doc_id') != did or chunk.get('sha256') != sha:
            raise ValueError('citation_source_version_mismatch')
        if q.get('doc_id',did) != did or q.get('sha256',sha) != sha:
            raise ValueError('citation_identity_mismatch')
        checked=registry.validate_answer({'status':'answered','claims':[{
            'doc_id':did,'category':raw['category'],'text':raw['text'],
            'citations':[{'chunk_id':q['chunk_id'],'quote':q.get('quote')}]}],
            'missing_information':[]},[chunk],allowed_doc_ids=[did])
        if not checked['valid']:
            raise ValueError('citation_literal_provenance_failed')
        base=checked['claims'][0]['citations'][0]
        start=q.get('quote_start',base['quote_start']);end=q.get('quote_end',start+len(q['quote']))
        if type(start)is not int or type(end)is not int:
            raise ValueError('citation_position_invalid')
        # A repeated string must stay bound to its original occurrence.
        binding={'chunk_id':chunk['chunk_id'],'char_start':start,'char_end':end,
            'quote_sha256':hashlib.sha256(q['quote'].encode()).hexdigest(),
            'alias':q.get('source_unit_alias','LOCAL')}
        cite=_binding_citation(registry,chunk,q['quote'],binding,base)
        for key in ('source_unit_mapping_fingerprint','source_request_fingerprint','source_origin'):
            if key in q:cite[key]=copy.deepcopy(q[key])
        if local:cite['source_origin']=raw.get('origin','local_source_extraction')
        fresh.append(cite)
    result={'doc_id':did,'category':FIELD_BY_ID[parsed['field_id']]['category'],
        'text':raw['text'],'citations':fresh,
        'origin':raw.get('origin','local_source_extraction' if local else 'validated_model_generation')}
    for key in ('extraction_rule','rule','source_context'):
        if key in raw:result[key]=copy.deepcopy(raw[key])
    return result


def _context_reason(claim, context):
    parsed=parse_field_claim(claim);fid=parsed['field_id'];text=parsed['text']
    quote='\n'.join(q['quote'] for q in claim['citations'])
    event=context.get('event_scope','unknown')
    role=context.get('evidence_role','unknown')
    warnings=context.get('warnings',[])
    if _normal(text) in _LABEL_ONLY:return 'heading_or_placeholder_is_not_fact'
    if _ABSENCE.search(text) or re.search(r'^(?:제출기한|제출방식|입찰 제출서류).{0,80}확인되지',text):
        return 'batch_absence_is_not_document_fact'
    if role=='other_project_candidate' or any('other_project' in str(w) for w in warnings):
        return 'different_project_in_source_appendix'
    if role=='example':return 'example_is_not_current_project_fact'
    if fid in {'project_name','issuing_agency','purpose','budget_vat','duration'} and _FORM_DESCRIPTION.search(text):
        return 'form_description_is_not_project_fact'
    if fid=='project_name':
        if role in {'source_form','agreement_clause'} or event in {'template','subcontract','briefing'}:
            return 'project_name_from_form_or_other_event'
        if re.search(r'하도급계약명|서식|목차|신청서|파일명',text):return 'project_name_from_form_or_filename'
        if text.count('_')>=2 and '제안요청서' in text:return 'filename_is_not_project_name'
    if fid=='issuing_agency':
        if _normal(text) in {'발주기관계약상대자','발주기관원도급자','운영기관계약상대자'}:
            return 'unfilled_party_labels_are_not_agency'
        if re.search(r'일시\s*및\s*장소|발표\s*관련|공고문',text):return 'event_schedule_is_not_agency'
        if re.fullmatch(r'[\w\s·()]+(?:그룹|팀|과|실)',text) and not re.search(r'공사|공단|협회|조합|재단|청|특별시|광역시|주식회사|\(주\)',text):
            return 'department_is_not_issuing_organization'
    if fid=='purpose':
        if role in {'source_form','agreement_clause'} or event in {'template','subcontract','briefing'}:
            return 'contract_or_form_purpose_is_not_project_purpose'
        if re.search(r'비밀유지계약|청렴계약|위탁자|수탁자|현장실사.*신청|사업목적.*항목|영향평가를.*실시',text):
            return 'contract_or_form_purpose_is_not_project_purpose'
        if re.search(r'제안요청서(?:이다|입니다)|교육계획.*제시|제안\s*특징.*기대효과',text):
            return 'document_description_or_proposal_instruction_is_not_purpose'
        if len(re.findall(r'(?:과업개요|과업목적|과업범위|과업내용)',text))>=3:
            return 'contents_list_is_not_project_purpose'
    if fid=='scope_requirements':
        if (re.search(r'파기(?:되었|하였)음을\s*확인|파기되었음을확인|상기와\s*같이.*확인하였',text,re.S)
            or re.search(r'^은\s{5,}.*을\s*수행',text,re.S)
            or re.fullmatch(r'[가-힣·\s]{3,45}(?:확인서|서약서|약정서)',text)):
            return 'blank_form_attestation_is_not_execution_fact'
    if fid=='budget_vat':
        if role=='source_form' or event in {'template','subcontract'}:
            return 'form_or_subcontract_value_is_not_project_budget'
        if re.search(r'실적\s*증명서|하도급\s*(?:계약|금액)|금액단위\s*[:：]|부가가치세\(면세\)',text):
            return 'form_or_historical_value_is_not_project_budget'
    if fid=='duration':
        if role in {'source_form','agreement_clause'} or event in {'template','subcontract','briefing'}:
            return 'other_contract_or_form_period'
        if re.search(r'처리기간|서식상|비밀유지의무|본 계약의 기간|하도급기간|위탁업무|개인정보',text):return 'other_contract_or_approval_period'
        if re.search(r'협상기간|협상대상자.{0,100}협상.{0,8}조정|소요비용.*가격견적서',text,re.S):
            return 'negotiation_or_cost_period_is_not_execution_duration'
    if fid in {'submission_documents','submission_deadline','submission_method'}:
        if event in {'briefing','execution','subcontract','template'}:
            return 'submission_event_is_not_tender'
        if role in {'agreement_clause','example'}:return 'contract_clause_is_not_tender_submission'
        if re.search(r'착수|준공|산출정보|산출물|사업완료|사업완료시|계약완료시|계약체결\s*시|반출입대장|실적\s*증명서.*신청|swit\.or\.kr',text,re.I):
            return 'post_contract_or_certificate_application'
        if fid=='submission_method' and re.search(r'입찰방식|계약문서에\s*포함|계약서와\s*동일|각\s*1통씩\s*보관',text):
            return 'procurement_type_or_contract_effect_is_not_delivery_route'
        if fid=='submission_method' and re.search(r'나라장터|G2B|전자조달',quote,re.I):
            if not re.search(r'입찰|가격|투찰|접수|제출|등록|장\s*소\s*[:：]',quote):
                return 'system_function_is_not_tender_delivery_route'
        if fid in {'submission_deadline','submission_method'} and event=='unknown' and not _NOTICE.search(quote):
            if not re.search(r'입찰|제안서|가격투찰|접수|전자조달|나라장터|G2B',quote,re.I):
                return 'tender_submission_context_not_established'
    if fid=='eligibility':
        if re.search(r'운영기관은\s*단일|사용자는\s*일반|경력기준.{0,100}점수|서류열람.*신청',text):
            return 'user_description_or_scoring_is_not_qualification'
    if fid=='evaluation':
        if re.search(r'(?:경우는|중에서|자를|자는|인자로|점수의)\s*$',text):
            return 'incomplete_evaluation_clause'
        actual_eval=re.search(r'기술성\s*평가\s*시.{0,70}하도급계획\s*적정성',quote,re.S)
        if (event in {'execution','subcontract','template'} or role in {'source_form','agreement_clause'}) and not actual_eval:
            return 'approval_or_execution_check_is_not_award_evaluation'
        if not actual_eval and re.search(r'하도급계약의?\s*적정성|하도급\s*적정성\s*판단|위약금|위규\s*수준|과업심의위원회|민간\s*소프트웨어\s*시장\s*침해',text):
            return 'approval_or_penalty_is_not_award_evaluation'
    return None


def _merge(claims):
    output=[];known={}
    for claim in claims:
        parsed=parse_field_claim(claim)
        key=(parsed['field_id'],parsed['mode'],_normal(parsed['text']))
        if key not in known:
            known[key]=copy.deepcopy(claim);output.append(known[key]);continue
        target=known[key];seen={(q['chunk_id'],q['quote_start'],q['quote_end']) for q in target['citations']}
        for q in claim['citations']:
            k=(q['chunk_id'],q['quote_start'],q['quote_end'])
            if k not in seen:target['citations'].append(copy.deepcopy(q));seen.add(k)
        origins=set(target.get('merged_origins',[target['origin']]))|{claim['origin']}
        target['merged_origins']=sorted(origins)
    return output


def _numeric_warnings(claims):
    warnings=[]
    for c in claims:
        parsed=parse_field_claim(c)
        if parsed['field_id']!='budget_vat':continue
        for match in re.finditer(r'(?<!\d)(\d{1,3}(?:,\d+)+)\s*원',parsed['text']):
            value=match.group(1)
            if any(len(p)!=3 for p in value.split(',')[1:]):
                warnings.append({'code':'irregular_source_money_grouping','field_id':'budget_vat',
                    'source_value':value,'message':'원문 금액의 자릿수 표기가 불규칙합니다. 공고 등으로 금액을 추가 확인하세요.',
                    'citations':copy.deepcopy(c['citations'])})
    return warnings


def rebuild_document_summary(result, document, chunks, registry):
    """Return a separately versioned source-based view; never mutate a run."""
    from .context import DocumentContext
    from .conditions import recover_conditions
    if not isinstance(result,dict) or not isinstance(document,dict):raise TypeError('문서와 결과는 dict여야 합니다.')
    did=document.get('doc_id');sha=document.get('sha256')
    if not did or result.get('doc_id')!=did or result.get('source_sha256')!=sha:
        raise ValueError('원문 ID·SHA와 생성 결과가 다릅니다.')
    if registry.source_hashes.get(did)!=sha:raise ValueError('등록된 원문 버전이 다릅니다.')
    canonical=list(registry.validate_chunks(chunks,allowed_doc_ids=[did]))
    by_id={c['chunk_id']:c for c in canonical}
    expected=result.get('source_chunk_ids',[])
    if not set(expected).issubset(by_id):raise ValueError('결과의 청크가 현재 원문에 없습니다.')
    context=DocumentContext(document)
    accepted=[];filtered=[];reclassified=[]
    raw_fingerprint=answer_fingerprint(result)

    def process(raw, *, local=False, rejection_id=None):
        try:
            c=_fresh_claim(raw,document,by_id,registry,local=local)
            ctx=context.classify_claim(c)
            reason=_context_reason(c,ctx)
            if reason:raise ValueError(reason)
            parsed=parse_field_claim(c)
            if _NOTICE.search('\n'.join(q['quote'] for q in c['citations'])) and parsed['field_id'].startswith('submission_'):
                c['text']=f'[{parsed["field_id"]}|notice_reference] '+parsed['text']
            c['content_context']={k:copy.deepcopy(ctx.get(k)) for k in
                ('event_scope','evidence_role','certainty','warnings','heading_path')}
            accepted.append(c)
            if rejection_id:reclassified.append(rejection_id)
        except (ValueError,TypeError,KeyError,AttributeError) as e:
            filtered.append({'reason':str(e) if isinstance(e,ValueError) else 'claim_format_invalid',
                'raw_claim':copy.deepcopy(raw),'rejection_id':rejection_id,
                'diagnostic_only':True})

    for c in result.get('claims',[]):process(c)
    complete=(result.get('execution_status') in {'completed','completed_with_rejections'}
        and result.get('input_chunk_coverage')==1.0 and result.get('checked_chunk_coverage')==1.0)
    # Do not turn a failed/unfinished model request into a completed answer.
    if complete:
        # A category typo does not establish entailment of a rejected model
        # paraphrase. Keep every rejected payload diagnostic-only. Important
        # omitted conditions are rediscovered below as exact source excerpts.
        for raw in recover_conditions(document,canonical,registry):
            raw=copy.deepcopy(raw)
            if 'field_id' in raw and not re.match(r'^\[',str(raw.get('text',''))):
                fid=raw.pop('field_id');mode=raw.pop('mode','fact')
                raw.update(doc_id=did,category=FIELD_BY_ID[fid]['category'],text=f'[{fid}|{mode}] '+raw['text'])
            raw.setdefault('origin','local_source_extraction')
            process(raw,local=True)
    claims=_merge(accepted)
    warnings=_numeric_warnings(claims)
    fields=[];missing=[]
    for spec in FIELDS:
        chosen=[]
        for c in claims:
            p=parse_field_claim(c)
            if p['field_id']==spec['id']:
                chosen.append({'text':p['text'],'mode':p['mode'],'citations':copy.deepcopy(c['citations']),
                    'origin':c['origin']})
        modes={c['mode'] for c in chosen}
        state=('conflict_candidates' if any(w['field_id']==spec['id'] for w in warnings) or 'conflict_candidate' in modes
               else 'notice_reference' if 'notice_reference' in modes and 'fact' not in modes
               else 'evidence_found' if chosen else 'not_found_in_processed_input' if complete else 'not_processed')
        notes=[w['message'] for w in warnings if w['field_id']==spec['id']]
        if not chosen:missing.append('['+spec['id']+'] '+('본문에서 확인하지 못했습니다.' if complete else '문서 처리가 완료되지 않았습니다.'))
        fields.append({'field_id':spec['id'],'label':spec['label'],'state':state,
            'claims':chosen,'notes':notes,'rejected_claim_count':0})
    # The existing notebook renderer displays missing_information, whereas its
    # per-field notes are diagnostic-only. Surface unresolved source typos too.
    missing.extend('['+w['field_id']+'] '+w['message'] for w in warnings)
    local_count=sum(c['origin'].startswith('local_source_') for c in claims)
    rebuilt=copy.deepcopy(result)
    rebuilt.update(content_version=CONTENT_VERSION,claims=claims,fields=fields,
        missing_information=missing,semantic_quality_verified=False,human_review_completed=False,
        all_documents_quality_verified=False,
        answer_generated=bool(result.get('answer_generated') and claims),
        generation_execution_status=result.get('execution_status'),
        generation_model_origin=copy.deepcopy(result.get('model_origin',{})),
        model_origin={'all_live_or_live_cache':bool(result.get('model_origin',{}).get('all_live_or_live_cache')) and not local_count,
            'derived_from_live_generation':bool(result.get('model_origin',{}).get('all_live_or_live_cache')),
            'contains_local_source_extraction':bool(local_count)},
        status=('partial' if missing or warnings else 'answered') if complete and claims else result.get('status','not_generated'))
    rebuilt['content_audit']={'version':CONTENT_VERSION,'raw_model_result_fingerprint':raw_fingerprint,
        'source_sha256':sha,'filtered_claims':filtered,'filtered_reason_counts':dict(Counter(r['reason'] for r in filtered)),
        'reclassified_rejection_ids':reclassified,'local_extraction_count':local_count,
        'original_rejected_claims_preserved':len(result.get('rejected_claims',[])),
        'output_claim_count':len(claims),'numeric_warnings':warnings,
        'raw_model_batch_notes':copy.deepcopy(result.get('missing_information',[])),
        'input_result_modified':False,'api_calls':0,'semantic_accuracy':None,
        'automatic_semantic_correctness_verified':False,'source_rules_are_semantic_oracle':False,
        'safe_counts':{'source_claims_validated':len(accepted),'output_claims':len(claims),
            'filtered_claims':len(filtered),'local_source_claims':local_count}}
    rebuilt['content_audit']['content_fingerprint']=_digest({
        'version':CONTENT_VERSION,'input':raw_fingerprint,'source':sha,
        'claims':claims,'fields':fields,'generation_execution_status':result.get('execution_status')})
    return rebuilt
