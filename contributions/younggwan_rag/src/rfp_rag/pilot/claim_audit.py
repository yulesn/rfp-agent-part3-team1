"""Literal requirement-name review hints, never an entailment/accuracy score.

Only accepted claims and their own quotations are compared. Names come from
requirement IDs in the selected source chunks, never a document-specific key.
Paraphrases, negation, numeric conditions and completeness require review even
when all literal anchors occur. The audit never edits or approves an answer.
"""
from __future__ import annotations

import html
import re


VERSION = 'phase8-claim-citation-audit-v1'
_ID = re.compile(r'(?<![\w])([A-Z]{2,8})\s*[-_]\s*'
    r'(?:([A-Za-z가-힣]{1,12})\s*[-_]\s*)?(\d{2,6})(?!\d)', re.I)
_HEADERS = {'요구사항고유번호', '요구사항분류응락수준', '요구사항명칭', '요구사항',
    '상세설명', '정의', '세부내용', '필수', '선택'}
_CLASS = re.compile(r'^(?:기능|성능|테스트|보안|품질|데이터|인터페이스|제약사항|'
    r'프로젝트관리|프로젝트지원)(?:필수|선택)?$')


def _key(value):
    value = re.sub(r'\b및\b', ' ', html.unescape(value).casefold())
    return re.sub(r'[\W_]+', '', value)


def _identity(match):
    return '-'.join(p.upper() for p in match.groups() if p)


def _catalog(chunks):
    catalog = {}
    for chunk in chunks:
        text = chunk.get('text', '')
        matches = list(_ID.finditer(text))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            for line in text[match.end():end].splitlines()[:6]:
                name = line.strip(' \t:|')
                compact = _key(name)
                if not compact or compact in _HEADERS or _CLASS.fullmatch(compact):
                    continue
                # A nearby heading is useful; a paragraph is not a name.
                if 2 <= len(compact) <= 60 and len(name) <= 100 and not compact.isdigit():
                    key = chunk['doc_id'], _identity(match)
                    catalog.setdefault(key, {})[compact] = name
                break
    return catalog


def claim_citation_audit(result):
    """Return bounded literal signals without borrowing other claims' sources."""
    accepted = (result.get('execution_status') == 'completed'
        and (result.get('validation') or {}).get('valid') is True)
    claims = (result.get('claims') or []) if accepted else []
    catalog = _catalog(result.get('source_chunks') or []) if claims else {}
    rows = []
    for i, claim in enumerate(claims):
        text = _key(claim['text'])
        declared_ids = {_identity(m) for m in _ID.finditer(claim['text'])}
        quotations = [(_key(c['quote']), {_identity(m) for m in _ID.finditer(c['quote'])})
            for c in claim.get('citations', [])]
        named = []
        for (did, requirement_id), names in catalog.items():
            if did != claim['doc_id']:
                continue
            mentioned = [key for key in names if key in text]
            if not mentioned and requirement_id not in declared_ids:
                continue
            candidates = mentioned or list(names)
            indices = [j for j, (quote, ids) in enumerate(quotations)
                if requirement_id in ids or any(key in quote for key in candidates)]
            named.append({'requirement_id': requirement_id,
                'label': names[candidates[0]], 'literal_citation_indices': indices})
        missing = [n for n in named if not n['literal_citation_indices']]
        rows.append({'claim_index': i, 'doc_id': claim['doc_id'],
            'citation_count': len(quotations), 'named_requirements': named,
            'missing_literal_anchors': [{'requirement_id': n['requirement_id'], 'label': n['label']}
                for n in missing],
            'signal': 'literal_anchor_gap' if missing else 'literal_anchors_found' if named else 'not_assessable'})
    return {'version': VERSION, 'accepted_claims_only': True, 'claim_count': len(rows),
        'assessable_claim_count': sum(bool(r['named_requirements']) for r in rows),
        'flagged_claim_indices': [r['claim_index'] for r in rows if r['missing_literal_anchors']],
        'rows': rows, 'semantic_correctness_verified': False, 'automatic_semantic_pass': False,
        'scope': 'Literal requirement-name/ID hints only. Missing anchors may be paraphrases; '
                 'present anchors do not prove meaning, conditions or completeness.'}


def claim_review_summary(audits):
    return {'version': VERSION, 'turns_with_literal_anchor_gaps': sum(bool(a['flagged_claim_indices']) for a in audits),
        'claims_with_literal_anchor_gaps': sum(len(a['flagged_claim_indices']) for a in audits),
        'assessable_claims': sum(a['assessable_claim_count'] for a in audits),
        'semantic_correctness_verified': False, 'automatic_semantic_pass': False}
