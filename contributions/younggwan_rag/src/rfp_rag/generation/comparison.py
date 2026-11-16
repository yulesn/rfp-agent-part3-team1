"""Document-local comparison requests and source-positioned duration roles.

These deterministic guards do not infer semantic correctness or convert time
units. Source quotations and immutable evidence identities remain unchanged.
"""
from __future__ import annotations

import re

from rfp_rag.retrieval.session import _project_title
from rfp_rag.conversation.resolution import title_variants, compact


def _identities(catalog):
    return {row['doc_id']:set(title_variants(_project_title(row)))|
            ({compact(row['doc_id'])} if len(compact(row['doc_id']))>=4 else set())
            for row in catalog}


def document_question(question, doc_id, catalog):
    """Retain requested fields/conditions, remove the other projects entirely."""
    rows={r['doc_id']:r for r in catalog}
    focus=re.sub(r'\n대상 사업:[^\n]*','',question).removeprefix('질문:').strip()
    aliases={v for values in _identities(catalog).values() for v in values}
    aliases.update(compact(r.get('agency') or '') for r in catalog)
    aliases.update(compact(r.get('file_name') or '') for r in catalog)
    for alias in sorted(aliases,key=len,reverse=True):
        if len(alias)<3:continue
        focus=re.sub(r'[\W_]*'.join(re.escape(c) for c in alias),' ',focus,flags=re.I)
    fields=re.search(r'비교\s*항목\s*(?:은|는|[:：])\s*(.*)',focus,re.S)
    if fields:focus=fields.group(1)
    focus=re.split(r'문서별\s*열|각\s*데이터\s*칸',focus,maxsplit=1)[0]
    focus=re.sub(r'「\s*[:：]?\s*」|『\s*[:：]?\s*』',' ',focus)
    focus=re.sub(r'선택한\s*(?:다음\s*)?두\s*(?:RFP|문서|사업)[^:：]*[:：]',' ',focus,flags=re.I)
    focus=re.sub(r'\s+',' ',focus).strip(' /:：;')
    return ('현재 분석 문서: '+_project_title(rows[doc_id])+'\n문서 ID: '+doc_id+
            '\n이 문서에서 확인할 항목과 조건: '+focus+
            '\n이 문서만 답하세요. missing_information에는 이 문서에서 확인하지 못한 요청 항목만 쓰세요. '
            '다른 비교 문서는 별도 요청에서 처리하므로 그 문서의 정보가 없다는 항목을 만들지 마세요. '
            '본문의 계약·과업기간과 별첨의 추정·적정 기간 산정자료를 구분하세요. '
            '별첨 수치는 산정자료라고 명시하고, 본문과 별첨을 한 주장에 합치거나 기간 단위를 환산하지 마세요.')


def local_missing_information(items, doc_id, catalog):
    """Exclude only unambiguous explicit references to another selected doc."""
    identities=_identities(catalog);kept=[];excluded=[]
    for item in items:
        value=compact(item);matches=[]
        for did,aliases in identities.items():
            for alias in aliases:
                for m in re.finditer(re.escape(alias),value):matches.append((did,m.start(),m.end()))
        maximal=[m for m in matches if not any(o[1]<=m[1] and m[2]<=o[2]
                   and o[2]-o[1]>m[2]-m[1] for o in matches)]
        mentioned={m[0] for m in maximal}
        if len(mentioned)==1 and doc_id not in mentioned:
            excluded.append({'item':item,'mentioned_doc_id':next(iter(mentioned)),
                             'reason':'other_selected_document_not_current_request'})
        else:kept.append(item)
    return kept,{'excluded_other_document_items':excluded,
                 'remaining_local_items':list(kept),'semantic_completeness_verified':False}


_ESTIMATE=re.compile(r'(?:적정|추정).{0,18}(?:사업|개발)\s*기간.{0,20}(?:산정서|산정표)|'
                     r'(?:사업|개발)\s*기간.{0,12}(?:종합\s*)?산정서')
_BOUNDARY=re.compile(r'^\s*(?:[<\[〈【]\s*(?:별첨|붙임|별지|서식)|'
                     r'[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[.．]?\s|\d+[.．]\s*\S)')


class DurationSources:
    """Classify exact quote ranges, including flattened PDF source blocks."""
    def __init__(self,document):
        self.lines={};self.roles={};self.headings={}
        estimate=None
        for collection in ('blocks','supplemental_blocks'):
            for block in document.get(collection,[]):
                key=collection,block['block_id'];lines=[];offset=0
                for raw in block['text'].splitlines(keepends=True):
                    text=raw.strip();start=offset;offset+=len(raw)
                    heading=(len(text)<200 and _ESTIMATE.search(text)
                             and not re.search(r'참조|참고|위와\s*같이|작성(?:해야|하여야)',text))
                    if heading:
                        estimate={'collection':collection,'block_id':block['block_id'],
                            'char_start':start,'char_end':offset,'text':text,'locator':block.get('locator')}
                    elif _BOUNDARY.search(text):estimate=None
                    lines.append((start,offset,estimate))
                self.lines[key]=lines

    def claim_role(self,claim):
        roles=set();headings=[]
        for citation in claim.get('citations',[]):
            for span in citation.get('source_ranges',[]):
                for start,end,heading in self.lines.get((span.get('collection','blocks'),span.get('block_id')),[]):
                    if max(start,span['char_start'])>=min(end,span['char_end']):continue
                    roles.add('estimate' if heading else 'other')
                    if heading and heading not in headings:headings.append(heading)
        role=('duration_mixed' if len(roles)>1 else 'duration_estimate'
              if roles=={'estimate'} else 'main_or_other_source')
        return {'role':role,'source_headings':headings,'semantic_correctness_verified':False}
