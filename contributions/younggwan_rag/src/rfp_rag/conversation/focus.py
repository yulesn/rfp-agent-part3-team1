"""User-intent hints only; never insert business facts or assert correctness."""
from __future__ import annotations

import re


WORK_SCOPE_PATTERN = (r'(?:사업|과업|공사|구축|개발)\s*(?:의\s*)?(?:범위|내용|대상)|'
    r'(?:어떤|무슨|무엇|뭘|뭐를).{0,30}(?:구축|개발|만들|만드|기능|업무)|'
    r'무슨\s*(?:일|작업)|뭐\s*하는\s*(?:사업|공사)|'
    r'(?:구축|개발).{0,20}(?:대상|기능)|대상\s*업무|업무\s*범위|기능\s*범위')
ACCEPTANCE_PATTERN = (r'(?:완료|준공|검수|인수)\s*(?:의\s*)?(?:조건|기준|요건|절차)|'
    r'준공|인수\s*(?:테스트|시험)|완료\s*검수|검수\s*(?:항목|방법)')
_FUNCTION_ID = re.compile(r'(?:SFR|FUN|FR)\s*[-_]\s*(?:[가-힣]+\s*[-_]\s*)?\d{2,}', re.I)
_SCOPE_HEADING = re.compile(r'(?:^|\n)\s*(?:[\d.가-힣]+[.)]\s*)?'
    r'(?:주요\s*)?(?:사업|과업|구축|개발)\s*(?:의\s*)?(?:범위|내용|대상)')
_SCOPE_TABLE = re.compile(r'대상\s*업무|업무\s*영역|주요\s*기능|요구사항\s*목록|기능\s*요구사항')
_GENERIC = re.compile(r'추진\s*목표|추진\s*전략|개발\s*방법론|개발\s*방향|최적화된?\s*방법|사업\s*이해')
_FORMS = re.compile(r'가격제안서|입찰서|실적증명서|보안약점설명|제안서평가|작성지침')


def focus_priority(chunk, topics, question):
    """Rank source candidates, preserving their literal content and provenance.

    Function IDs and scope headings can expose concrete work items; they do
    not prove that those items answer the user or belong to this contract.
    Plain construction scopes are supported without any IT/business names.
    """
    if not {'work_scope', 'acceptance'}.intersection(topics):
        return 0, 0
    text = chunk['text']
    compact = re.sub(r'\s+', '', text)
    form_requested = bool(re.search(r'서식|양식|가격\s*제안서|입찰서|보안약점', question))
    form_penalty = bool(_FORMS.search(compact)) and not form_requested
    score = 0
    if 'work_scope' in topics:
        count = len(_FUNCTION_ID.findall(text))
        score += (4 + min(count, 4)) if count else 0
        score += 3 * bool(_SCOPE_HEADING.search(text)) + 2 * bool(_SCOPE_TABLE.search(text))
        score -= 2 * bool(_GENERIC.search(text))
    if 'acceptance' in topics:
        score += 2 * bool(re.search(r'인수\s*테스트|인수시험|검수|검사\s*완료', text))
        score += bool(re.search(r'산출물|완료보고|운영\s*전환|기술\s*이전', text))
    return -int(form_penalty), score


def focused_question(question, topics):
    """Versioned, fact-free instructions; exact question cache keys change."""
    instructions = []
    if 'work_scope' in topics:
        instructions.append('사용자는 구체적인 사업·과업의 내용과 구축 대상을 묻습니다. '
            '제공된 근거의 실제 대상 업무·기능·구축 범위를 구분해 직접 답하세요. '
            '개발 방향을 제시하라는 일반 원칙이나 제안서 작성 지침만으로 구체적인 범위를 대신하지 마세요. '
            '기존 시스템 현황, 이번 사업 범위, 검수 후 지원 의무를 서로 구분하고, '
            '문서에 없는 세부 내용은 missing_information에 남기세요.')
    if 'acceptance' in topics:
        instructions.append('검수·인수·완료 조건과 검수 후의 보고·하자보수·지원 의무를 구분하세요. '
            '완료 후 제출 기한이나 지원 기간을 완료 이전의 선행 조건으로 바꾸지 마세요.')
    return question + ('\n\n질문 초점 안내(focus-v1): ' + ' '.join(instructions) if instructions else '')


def response_focus_audit(result, topics):
    """A review hint, never a semantic pass or a change to accepted claims."""
    claims = result.get('claims') or []
    generic = [i for i, claim in enumerate(claims) if _GENERIC.search(claim.get('text', ''))]
    return {'version': 'phase6-focus-audit-v1', 'requested_topics': list(topics),
        'generic_process_claim_indices': generic,
        'needs_scope_review': bool('work_scope' in topics and generic),
        'semantic_correctness_verified': False,
        'note': 'Keyword hints only. Check actual work items, scope, completeness and citations manually.'}
