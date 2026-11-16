"""Bound requested award facts to the evidence actually supplied this turn.

This is a conservative retrieval gate, not a semantic correctness evaluator.
Missing award evidence never becomes a claim that an award does not exist.
"""
from __future__ import annotations

import re


_AWARD = re.compile(r'최종\s*낙찰\s*(?:업체|기업|회사|자)(?!\s*(?:선정|결정|평가|기준|방식|방법))|'
                    r'낙찰\s*(?:받은|된)\s*(?:업체|기업|회사)|누가.{0,10}낙찰')
_ACTUAL_AMOUNT = re.compile(r'실제\s*(?:계약|낙찰)\s*(?:금액|가격)|'
                            r'(?:체결된|체결한)\s*계약\s*금액|최종\s*낙찰\s*(?:금액|가격)')
_GENERAL = re.compile(r'예산|사업비|용역비|부가세|부가가치세|VAT|기간|개월|자격|'
                      r'제출|요구사항|선정|평가|기준|방식|방법|보증|사업범위', re.I)
_AWARD_VALUE = re.compile(r'(?:(?:최종\s*)?낙찰\s*(?:업체(?:명)?|기업|회사)\s*(?:[:：]|은|는)|'
                          r'최종\s*낙찰자\s*(?:[:：]|은|는)|낙찰자\s*[:：])\s*'
                          r'([가-힣A-Za-z0-9㈜][^\n]{1,100})')
_AMOUNT_VALUE = re.compile(r'(?:실제\s*계약|체결\s*계약|최종\s*낙찰|낙찰)\s*(?:금액|가격)\s*(?:[:：]|은|는)\s*'
                           r'(?:금\s*)?\d[\d,.]*\s*(?:억|백만|천만|천|만)?\s*원')


def award_evidence_gate(question, chunks):
    """Withhold an exclusive result question lacking explicit result values.

    Mixed questions still reach generation: an RFP budget can answer a budget
    sub-question, but cannot substitute for an actual awarded contract amount.
    Even explicit result candidates must pass the normal generation/citation
    checks; this gate never emits source facts or labels a candidate correct.
    """
    fields=[]
    if _AWARD.search(question):fields.append('최종 낙찰업체')
    if _ACTUAL_AMOUNT.search(question):fields.append('실제 계약금액')
    remaining=_ACTUAL_AMOUNT.sub(' ',_AWARD.sub(' ',question))
    if '실제 계약금액' in fields:
        remaining=re.sub(r'부가세|부가가치세|VAT',' ',remaining,flags=re.I)
    if not fields or _GENERAL.search(remaining):return None
    text='\n'.join(c['text'] for c in chunks)
    candidates=[]
    if '최종 낙찰업체' in fields:
        candidates += [m.group(1).strip() for m in _AWARD_VALUE.finditer(text)
                       if not re.match(r'선정|미정|미확정|미기재|해당\s*없|없음|공란',m.group(1))]
    if '실제 계약금액' in fields and _AMOUNT_VALUE.search(text):candidates.append('amount_candidate')
    if candidates:return None
    return {'policy':'requested-award-evidence-v1','requested_fields':fields,
            'reason':'requested_result_not_in_supplied_evidence',
            'scope':'Current retrieved input only; not proof of absence from all documents.',
            'additional_materials':['낙찰 결과 공고','체결 계약서'],
            'semantic_correctness_verified':False}


def generation_question(question, *, doc_id=None):
    instructions=('질문에서 요청한 항목에 직접 답하는 주장만 작성하세요. '
        '배경 예산·평가·보증 조건을 요청한 낙찰 결과 대신 답하지 마세요. '
        '과업예산과 실제 계약금액을 구분하고, 요청 항목의 근거가 모두 부족하면 '
        'claims=[]인 insufficient_evidence로 답하세요. '
        '인용은 원문 그대로 복사하고 서로 떨어진 문장을 / 또는 ...로 이어 붙이지 마세요.')
    if doc_id is not None:
        instructions += (' 이번 요청은 비교표의 한 문서만 처리합니다. '
            f'모든 claim.doc_id는 {doc_id}여야 합니다. '
            '다른 비교 문서에 관한 주장은 쓰지 말고 이 문서의 요청 항목만 추출하세요.')
    return question+'\n\n답변 작성 조건: '+instructions
