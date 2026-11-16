"""Recover explicit RFP conditions from the registered source, without a model.

This is a precision-oriented, verbatim supplement to a model's extraction. It
does not fill gaps from a filename, reference answer, external knowledge, or
an unregistered OCR view. Every emitted character has canonical source ranges.
The caller must still apply document context and review the resulting summary.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re

from rfp_rag.summary_fields.fields import FIELD_BY_ID

VERSION = 'phase5-explicit-source-conditions-v1'


@dataclass(frozen=True)
class SourceLine:
    collection: str
    block_id: str
    start: int
    end: int
    text: str
    locator: str


def _compact(value):
    return re.sub(r'\s+', '', value)


def _lines(document):
    """Split only at real source newlines; retain original block coordinates."""
    output = []
    for collection in ('blocks', 'supplemental_blocks'):
        for block in document.get(collection, []):
            if collection == 'supplemental_blocks' and not (
                block.get('source_method') == 'reviewed_visual_transcription'
                and block.get('review_status') == 'visual_verified'
            ):
                continue
            text = block.get('text', '')
            position = 0
            for piece in text.splitlines(keepends=True):
                end = position + len(piece.rstrip('\r\n'))
                if text[position:end].strip():
                    output.append(SourceLine(collection, block['block_id'], position,
                        end, text[position:end], block['locator']))
                position += len(piece)
    return output


def _range_index(chunks):
    result = defaultdict(list)
    for chunk in chunks:
        for span in chunk['source_spans']:
            result[(span['collection'], span['block_id'])].append((span, chunk))
    return result


def _citation(registry, chunk, left, right, did):
    quote = chunk['text'][left:right]
    payload = {'status':'answered', 'claims':[{'doc_id':did, 'category':'other',
        'text':'원문 직접 발췌', 'citations':[{'chunk_id':chunk['chunk_id'], 'quote':quote}]}],
        'missing_information':[]}
    checked = registry.validate_answer(payload, [chunk], allowed_doc_ids=[did])
    if not checked['valid']:
        raise ValueError('Recovered source quote failed provenance validation: ' + '; '.join(checked['errors']))
    resolved = checked['claims'][0]['citations'][0]
    # A quote may repeat inside one chunk. Preserve the selected physical range,
    # rather than validate_answer's first text match.
    ranges = []
    covered = []
    for span in chunk['source_spans']:
        start, end = max(left, span['chunk_start']), min(right, span['chunk_end'])
        if start >= end:
            continue
        row = {'collection':span['collection'], 'block_id':span['block_id'],
            'locator':span['locator'],
            'char_start':span['char_start'] + start - span['chunk_start'],
            'char_end':span['char_start'] + end - span['chunk_start'],
            'chunk_start':start, 'chunk_end':end}
        for key in ('page','section','table_id','row','col'):
            if key in span:
                row[key] = span[key]
        ranges.append(row)
        covered.append((start-left, end-left))
    cursor = 0
    for start,end in sorted(covered):
        if quote[cursor:start].strip():
            raise ValueError('Recovered citation contains uncovered text')
        cursor = max(cursor,end)
    if not ranges or quote[cursor:].strip():
        raise ValueError('Recovered citation has no complete source coverage')
    resolved.update(quote_start=left, quote_end=right, source_ranges=ranges,
        locator=' / '.join(dict.fromkeys(r['locator'] for r in ranges)),
        source_position_verified=True, source_origin='local_source_extraction')
    return resolved


def _citations_for(lines, intervals, registry, did):
    citations = []
    seen = set()
    for line in lines:
        cursor = line.start
        choices = intervals.get((line.collection,line.block_id), [])
        while cursor < line.end:
            spans = [(span,chunk) for span,chunk in choices
                if span['char_start'] <= cursor < span['char_end']]
            if not spans:
                # Whitespace omitted by a parser/chunker is not evidence text.
                if not line.text[cursor-line.start:].strip():
                    break
                return []
            span,chunk = max(spans,key=lambda pair:pair[0]['char_end'])
            stop = min(line.end,span['char_end'])
            left = span['chunk_start'] + cursor-span['char_start']
            right = left+stop-cursor
            if not chunk['text'][left:right].strip():
                cursor=stop
                continue
            key = chunk['chunk_id'],left,right
            if key not in seen:
                citations.append(_citation(registry,chunk,left,right,did))
                seen.add(key)
            cursor = stop
    return citations


_NUMBER = re.compile(r'\d')
_DATE = re.compile(r'(?:\d[\d.\s~∼～\-]*[년월일]|\d[\d,.\s]*개월|계약\s*체결|계약일|계약\s*시작)')
_FORM = re.compile(r'(?:서식\s*\d|<\s*서식|작성\s*(?:예시|요령)|기재\s*(?:예시|요령)|단순\s*예시|서식상|하도급\s*(?:금액\s*산출|계약\s*가격)|사업\s*수행\s*실적\s*증명)')
_ABSENT = re.compile(r'(?:확인되지\s*않|확인할\s*수\s*없|명시되어\s*있지\s*않)')
_NOTICE = re.compile(r'(?:공고문|입찰공고|입찰유의서).{0,40}(?:참조|참고|의함|따르|따름|명기|확인)')
_BID = re.compile(r'입찰|제안서|가격\s*(?:투찰|입찰|제안)|공고문|입찰유의서')
_EXEC = re.compile(r'착수계|준공계|준공검사|준공정산|착수\s*(?:시|후|및)|준공\s*(?:시|후)|결과보고서|주간보고서|월간보고서')
_BRIEF = re.compile(r'사전\s*설명회|설명회\s*(?:참가|참석|신청)|현장\s*실사\s*참가')


def _primary_section(lines, i):
    """Find explicit nearby purpose/scope headings, stopping at new sections.

    Korean RFP purpose bullets often use ``1)`` themselves. They are facts
    under a purpose heading, not a reason to discard the paragraph.
    """
    for line in reversed(lines[max(0,i-65):i+1]):
        text=line.text.strip();dense=_compact(text)
        if re.search(r'\t\s*\d+\s*$|[·.…]{3,}\s*\d+\s*$',text):
            continue
        # A sentence mentioning an expected effect (especially an evaluation
        # rubric) is not a purpose heading. Only explicit section labels open it.
        heading = re.sub(r'^(?:[□■○ㅇo\d.()ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|[가-하]\.)*', '', dense)
        inline_purpose = bool(re.match(r'^[□■○ㅇo\d.]*\((?:추진목적|기대효과)\)',dense))
        if (heading in {'사업목적','과업목적','추진목적','추진배경및목적','추진배경및필요성','기대효과'}
            or inline_purpose):
            if not re.search(r'목차|서식|제안서작성|작성요령|하도급',dense):
                return 'purpose'
        if len(dense)<45 and re.search(r'(?:주요사업내용|주요사업범위|과업범위|과업내용|사업내용및범위)$',dense):
            return 'scope_requirements'
        # An actual numbered/marked section ends the preceding semantic scope.
        if len(dense)<90 and (
            re.match(r'^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.?|\d{1,2}\.)(?:사업|과업|입찰|제안|수행|참가|평가|선정|일반|개요|보안|위반|비밀|기타|용역|추진)',dense)
            or re.match(r'^[가-하]\.\s*',text)
            or re.match(r'^\d{1,2}\s+(?:사업개요|개요|조직체계|추진일정|사업내용)',text)
            or re.match(r'^[□■](?:용역명|사업명|계약방법|과업예산|추진일정|참가자격)',dense)
            or re.match(r'^(?:요구사항(?:명칭|고유번호|분류)|평가항목|평가요소|제안서목차|목차)$',dense)):
            return None
    return None


def _matches(lines, i):
    """Return (field, rule, mode, required context lines) for explicit language."""
    line = lines[i]
    text = line.text
    dense = _compact(text)
    preceding = lines[max(0,i-4):i]
    # A new actual heading ends the previous event. In particular, the same
    # email address may be used first for a briefing and later for the bid.
    for j in range(len(preceding)-1,-1,-1):
        value=preceding[j].text.strip()
        if len(value)<100 and re.match(r'^(?:[□■]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[.、]?|[IVX]{1,5}\.\s|[가-하]\.\s)',value):
            preceding=preceding[j:]
            break
    previous = '\n'.join(v.text for v in preceding)
    nearby = previous+'\n'+text+'\n'+'\n'.join(v.text for v in lines[i+1:i+3])
    event_context = previous+'\n'+text
    is_form = bool(_FORM.search(text))
    out = []

    def add(field, rule, mode='fact', include_previous=0, include_next=0):
        start=max(0,i-include_previous);end=min(len(lines),i+include_next+1)
        out.append((field,rule,mode,lines[start:end]))

    def paragraph(field,rule):
        count=0
        # PDF source pages wrap a paragraph over physical lines. Include only
        # continuation lines from that same block, stopping at a fresh bullet.
        for nxt in lines[i+1:i+5]:
            if nxt.block_id!=line.block_id or nxt.collection!=line.collection:
                break
            value=nxt.text.strip()
            if (re.match(r'^(?:[□■○ㅇo※∙­-]|\d+[.)]|[가-하]\.\s|[ⅠⅡⅢⅣⅤ])',value)
                or re.match(r'^\d{1,2}\s+(?:사업|과업|추진|개요|일반|입찰|제안|조직)',value)):
                break
            count+=1
        add(field,rule,include_next=count)

    # Actual headline pairs; only a concrete value is emitted, never an empty
    # "사업명" or "발주기관" form label. Context filtering rejects appendices.
    for field,pattern in [('project_name',r'(?:사업명|과업명|용역명)'),
                          ('issuing_agency',r'(?:발주기관|주관기관|운영기관)')]:
        if re.fullmatch(r'[□○ㅇo\d.()\s]*(?:'+pattern+r')\s*[:：]?',dense):
            if i+1<len(lines) and len(lines[i+1].text.strip())>2:
                add(field,'explicit_heading_value',include_next=1)
        elif re.match(r'^[□○ㅇo\d.()\s]*(?:'+pattern+r')[:：]',dense):
            if dense.split(':',1)[-1].strip() and not is_form:
                add(field,'explicit_heading_inline')

    if re.search(r'(?:사업예산|과업예산|용역비|사업비|예정가격|추정가격)',dense) and re.search(r'\d[\d,]*(?:원|백만|억)',dense) and not is_form:
        add('budget_vat','project_budget')
    if re.search(r'개발예산|(?:HW|H/W|하드웨어)(?:구입|구매|예산)',dense,re.I) and _NUMBER.search(text) and not is_form:
        add('budget_vat','budget_breakdown')
    if re.search(r'단가계약|실제.{0,15}(?:인원|대상|물량|제공량).{0,30}정산|대금지급금액.{0,40}제공량',dense):
        add('budget_vat','actual_quantity_settlement')
    if re.search(r'시간당정산단가|사용시간은.{0,12}단위|예산한도.{0,20}(?:도달|축소)|정산금액.{0,50}예산',dense):
        add('budget_vat','usage_settlement_condition')
    if re.search(r'제안가격.{0,30}(?:모든세금|세금.{0,10}포함)',dense):
        add('budget_vat','all_taxes_included')
    if re.search(r'사업예산.{0,20}사업기간.{0,50}조정',dense):
        add('budget_vat','budget_adjustment')
        add('duration','duration_adjustment')

    if re.search(r'(?:사업기간|과업기간|용역기간)',dense) and _DATE.search(text) and not is_form:
        add('duration','project_duration')
    if re.match(r'^[\d)\s]*기간[:：]',dense) and _DATE.search(text) and re.search(r'과업기간|사업개요',_compact(previous)):
        add('duration','duration_under_project_heading')
    if re.search(r'상반기|하반기',dense) and _DATE.search(text):
        add('duration','phased_program_dates')
    if (re.search(r'(?:과업기간|사업기간|계약기간).{0,20}(?:이후|종료후)|(?:사업|과업)종료이후|과업종료후정산',dense)
        and not re.search(r'비밀유지|위탁업무|개인정보|제3자.{0,15}(?:누설|제공)',dense)):
        add('duration','post_completion_duty')
    if re.search(r'(?:일정|기간|종료일).{0,40}(?:조정|연장|축소)|최대\d+개월|예산소진시',dense) and not is_form and re.search(r'사업|과업|계약|자원|제공|추진',nearby):
        add('duration','duration_exception')
    if re.search(r'(?:협의|연계|승인).{0,15}지연.{0,35}(?:조정|단계적)',dense) and re.search(r'사업기간|과업기간|계약',_compact(previous)):
        add('duration','coordination_delay_exception')
    if re.search(r'준비.{0,25}시간.{0,20}(?:미포함|제외)|미소진.{0,40}(?:추가제공|크레딧)',dense):
        add('duration','resource_time_condition')
    if re.search(r'하자보수기간.{0,35}(?:검수|준공).{0,20}\d+년',dense):
        add('duration','separate_warranty_period')

    if re.search(r'업종코드|직접생산(?:확인)?증명|중소기업.{0,20}확인서|대기업.{0,30}참여.{0,10}제한|상호출자제한기업집단',dense) and not is_form:
        add('eligibility','business_eligibility')
    if (re.search(r'대기업|중견기업',dense) and re.search(r'참여할수없|참여불가|참여제한',dense)
        and not is_form):
        add('eligibility','business_participation_prohibition')
    if re.search(r'공동(?:수급|도급).{0,40}(?:\d+개|허용|불허)|공동이행방식만|최소지분율',dense) and not is_form:
        add('eligibility','joint_venture_conditions')
    if re.search(r'소프트웨어사업자|컴퓨터관련서비스사업|보안인증\(CSAP|CSAP\)',dense) and re.search(r'자격|신고|등록|소지|획득|대상|필요',dense) and not is_form:
        add('eligibility','registered_business_or_security_qualification')
    if re.search(r'(?:참여인력|근무인력|총괄책임자|사업관리자\(PM\)).{0,80}(?:경력|유경험자|계속근로|상주|정규직)',dense) and not is_form:
        add('eligibility','performance_staff_requirement')
    if i+1<len(lines) and lines[i+1].block_id==line.block_id and not is_form:
        continuation_dense=dense+_compact(lines[i+1].text)
        if (re.search(r'소프트웨어사업자|컴퓨터관련|직접생산|대기업|중견기업',dense)
            and re.search(r'신고|등록|소지|참여할수없|참여제한',continuation_dense)):
            paragraph('eligibility','wrapped_business_qualification')
        if (re.search(r'참여인력|근무인력|총괄책임자|사업관리자\(PM\)',dense)
            and re.search(r'(?:참여인력|근무인력|총괄책임자|사업관리자\(PM\)).{0,120}(?:유경험자|계속근로|상주|정규직)',continuation_dense)):
            paragraph('eligibility','wrapped_staff_requirement')

    if _NOTICE.search(text) and _BID.search(nearby) and not _EXEC.search(text) and not _BRIEF.search(event_context):
        if re.search(r'기한|일시|일정|제출',dense):
            add('submission_deadline','external_notice_deadline','notice_reference')
        if re.search(r'입찰서류|제출서류|제안서제출',dense):
            add('submission_documents','external_notice_documents','notice_reference')
        if re.search(r'방법|장소|제출|입찰서류',dense):
            add('submission_method','external_notice_method','notice_reference')
    if (re.search(r'국가종합전자조달|나라장터|G2B',text,re.I)
        and (re.search(r'가격|입찰|등록|투찰|접수|제출',text)
             or re.search(r'장\s*소\s*[:：]',text) and re.search(r'가격|입찰|등록|투찰',previous))
        and not _EXEC.search(text)):
        add('submission_method','electronic_tender_price')
    if re.search(r'우편|이메일|전자우편|직접방문|방문제출|방문접수',dense) and _BID.search(nearby) and not _BRIEF.search(event_context) and not _EXEC.search(text):
        add('submission_method','tender_delivery_method',include_next=1 if ('사전협의' in ''.join(v.text for v in lines[i+1:i+2])) else 0)
    if re.search(r'입찰금액.{0,40}동일|전자입찰.{0,35}무효|우편제출.{0,50}도착분',dense):
        add('submission_method','tender_delivery_validity')
    if re.search(r'평가용.{0,45}(?:기재하지|빈칸|사용불가)|밀봉.{0,20}날인|USB.{0,30}(?:인쇄본|PPT|PDF)|인쇄본.{0,20}불필요',dense) and not _EXEC.search(nearby) and not is_form:
        add('submission_documents','tender_copy_anonymity_sealing')
    if re.search(r'발표자료.{0,30}사전제출',dense) and not is_form:
        paragraph('submission_documents','presentation_materials_before_evaluation')
    if ('신용평가등급확인서' in dense and '제출' in _compact(nearby)
        and not _EXEC.search(text) and not is_form):
        paragraph('submission_documents','tender_credit_rating_certificate')

    # Evaluation percentages always carry the threshold's referent, avoiding
    # approval (e.g. subcontract approval) or penalties mistaken for bid scores.
    if re.search(r'협상적격|기술능력.{0,25}평가|기술평가|종합평가점수|낙찰자선정방식',dense) and not is_form and not re.search(r'하도급계약.{0,30}(?:승인|적정성)',dense):
        if re.search(r'\d+(?:\.\d+)?(?:점|%)|적격심사|낙찰제',dense):
            continuation = _compact(''.join(v.text for v in lines[i+1:i+2]))
            add('evaluation','evaluation_ratio_or_threshold',include_next=1
                if re.search(r'(?:경우는|자로|자가|인자를|자중에서)$',dense)
                and re.search(r'협상적격자|협상대상자',continuation) else 0)
    if (re.search(r'기술능력|계량부분|비계량',dense) and re.search(r'\d+(?:\.\d+)?점',dense)
        and '평가' in dense and not is_form):
        add('evaluation','quantitative_and_qualitative_weights')
    if re.search(r'합산점수가동일|기술능력평가점수도동일|배점이동일한평가항목|최고점수.{0,30}최저점수',dense) and not is_form:
        add('evaluation','evaluation_tie_or_trim')
    if re.search(r'차순위협상적격자.{0,25}협상',dense) and not is_form:
        add('evaluation','next_ranked_negotiation')
    if re.search(r'기술성평가시.{0,40}하도급계획적정성',dense):
        add('evaluation','tender_subcontract_plan_evaluation')
    if re.search(r'평가항목에포함.{0,100}확인서.{0,20}제출',dense) and '하도급계획적정성' in dense:
        add('submission_documents','subcontract_plan_for_tender')
    if re.fullmatch(r'\(?\d+(?:\.\d+)?점\)?|\(\d+(?:\.\d+)?\)',dense):
        # Flattened HWP/HWPX tables put a label and its parenthesized score in
        # separate blocks. Retain the exact cell sequence, never infer a sum.
        labels=[]
        for prior in reversed(lines[max(0,i-4):i]):
            label=prior.text.strip()
            if not label or len(_compact(label))>24 or re.search(r'\d|[。.!?]|^[○ㅇo※∙­-]',label):
                break
            labels.append(prior)
        labels=list(reversed(labels))
        labeltext=_compact(''.join(v.text for v in labels))
        preceding_eval=''.join(v.text for v in lines[max(0,i-100):i])
        if (labels and re.search(r'평가|능력|방법론|관리|지원',labeltext)
            and re.search(r'평가\s*(?:항목|방법)|제안서\s*평가',preceding_eval)):
            add('evaluation','evaluation_table_label_and_score',include_previous=len(labels))

    if re.search(r'(?:최소|이상|이내|정확도|예정인원|예상건수|모델).{0,35}\d|\d.{0,25}(?:단위|이상|이내|정확도|최소)',dense) and re.search(r'수집|정확도|GPU|예정인원|발급예상건수|보유|할당',text,re.I) and not is_form:
        add('scope_requirements','minimum_or_estimated_quantity')
    if re.search(r'모델.{0,50}\d+(?:\.\d+)?%.{0,25}이상',dense) and re.search(r'GPU|자원|가속',nearby,re.I) and not is_form:
        add('scope_requirements','resource_model_ratio')
    if re.search(r'정확도산출.{0,40}방법론|알고리즘.{0,25}교체|수집이?지속가능|국내보유|주\d+회이상.{0,40}방문',dense):
        add('scope_requirements','critical_execution_condition')
    if re.search(r'예정인원.{0,35}(?:변동|추정|증감)|실제대상인원.{0,25}증감',dense):
        add('scope_requirements','estimated_quantity_variation')
    if re.search(r'(?:사용량|사용률|이용량|이용률).{0,50}(?:모니터링|통계)|모니터링.{0,50}(?:사용량|사용률)',dense):
        paragraph('scope_requirements','resource_usage_monitoring')
    following=_compact(''.join(v.text for v in lines[i:i+2]))
    if (re.search(r'현재.{0,20}\d+개.{0,100}시스템',following)
        and re.search(r'시스템.{0,10}운영',following)):
        paragraph('scope_requirements','named_current_system_scope')
    if re.search(r'협의에따라.{0,20}(?:세부사항|장소).{0,20}(?:변동|결정)|협의에따라세부사항변동',dense) and not is_form:
        add('scope_requirements','scope_by_agreement')
    if re.search(r'사업종료이후에도.{0,50}자료보완',dense):
        add('scope_requirements','post_completion_information_support')
    if re.search(r'개인정보.{0,40}(?:반납|파기)|자료.{0,30}(?:완전폐기|삭제.{0,15}반납)',dense) and not is_form:
        add('scope_requirements','data_return_or_destruction')

    section=_primary_section(lines,i)
    if section=='purpose' and len(dense)>8 and not is_form and not _ABSENT.search(text):
        if (not re.search(r'(?:목차|서약|약정|비밀유지|개인정보.{0,8}계약)',text)
            and re.search(r'효율|확보|개선|개편|안정|지원|제고|최소|운영|설계|마련|분석|제공|조성|재정의|활용|도모|필요|구축|전환|인지도|대응|촉진|확대',text)):
            paragraph('purpose','purpose_under_explicit_heading')
    if section=='scope_requirements' and len(dense)>22 and not is_form:
        if re.search(r'운영|수행|구축|제공|분석|지원|통합|데이터|인프라|사업',text):
            paragraph('scope_requirements','scope_under_explicit_heading')
    return out


def recover_conditions(document, chunks, registry):
    """Return canonical, source-only claims for explicit essential conditions.

    Output uses the established [field_id|mode] transport prefix. The origin is
    always local_source_extraction and is never represented as a model answer.
    Rules produce candidates; contextual rejections remain visible to callers
    in their own processing diagnostics. There is no per-document ID exception.
    """
    did = document.get('canonical_doc_id',document['doc_id'])
    if document['doc_id'] != did:
        raise ValueError('Condition recovery requires the canonical document')
    selected = [c for c in chunks if c.get('doc_id') == did]
    if not selected:
        raise ValueError('Condition recovery requires registered source chunks')
    selected = registry.validate_chunks(selected,allowed_doc_ids=[did])
    if any(c.get('sha256') != document.get('sha256') for c in selected):
        raise ValueError('Condition recovery source versions differ')
    lines = _lines(document)
    intervals = _range_index(selected)
    output = []
    seen = set()
    for i,line in enumerate(lines):
        for field,rule,mode,parts in _matches(lines,i):
            value = '\n'.join(v.text for v in parts)
            if not value.strip() or _ABSENT.search(value) or re.search(r'\t\s*\d+\s*$|[·.…]{3,}\s*\d+\s*$',value):
                continue
            citations = _citations_for(parts,intervals,registry,did)
            if not citations:
                continue
            identity = field,mode,value
            if identity in seen:
                continue
            seen.add(identity)
            output.append({'doc_id':did,'category':FIELD_BY_ID[field]['category'],
                'text':f'[{field}|{mode}] '+value, 'citations':citations,
                'field_id':field,'mode':mode,'origin':'local_source_extraction',
                'extraction_rule':rule,'extraction_version':VERSION,
                'source_text_sha256':hashlib.sha256(value.encode('utf-8')).hexdigest(),
                'semantic_quality_verified':False})
    return output
