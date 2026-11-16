"""Constrain work-scope citations to exact, chunk-bound source excerpts.

This prevents a model from putting an ellipsis-joined outline in a quotation.
It never repairs such an outline, drops a claim, or weakens EvidenceRegistry.
The same guarded client, counters, cache and request caps remain in use.
"""
from __future__ import annotations

import copy

from rfp_rag.generation.evidence import ANSWER_SCHEMA
from rfp_rag.generation.llm import COMPARISON_DOCUMENT_TASK, LLMError


VERSION = 'work-scope-literal-quotes-v3'
QA_TASK = 'qa_work_scope_literal_v3'
MAX_QUOTE_BYTES = 420
MAX_OPTIONS = 256
GUIDE = ('인용 선택 규칙(literal-quotes-v3): 각 evidence의 citation_options는 해당 원문의 '
    '정확한 인용 선택지입니다. 응답 citations의 quote에는 원문 대신 해당 선택지의 ref '
    '(예: q0001)만 넣고 그 선택지가 속한 chunk_id를 함께 지정하세요. 서버가 ref를 '
    '정확한 원문 인용으로 변환한 뒤 문서와 청크 연결을 검사합니다. 여러 표 제목을 ... 또는 /로 '
    '이어 쓴 요약문을 인용문으로 만들지 마세요. 필요한 근거가 떨어져 있으면 별도 citations로 '
    '선택하세요. 구체적인 대상 업무·기능을 중심으로 핵심 내용을 정리하고 일반 성능·보안·평가 '
    '목록을 사업의 구체적인 대상 업무 대신 나열하지 마세요. 각 주장의 숫자·조건도 선택한 '
    '인용에 포함되어야 하며, 근거가 부족한 항목은 missing_information에 남기세요. '
    '각 claims 항목은 핵심 업무·기능별로 나누세요. 한 문장에 여러 요구사항 명칭을 나열하면 '
    '그 항목 모두를 바로 그 문장의 citations로 뒷받침하세요. 다른 claims 항목에 붙인 인용을 '
    '이 문장의 근거로 대신하지 마세요. 필요한 선택지가 입력에 없으면 주장을 좁히거나 '
    '미확인 항목으로 남기고, 관련 없는 검수 절차를 구체적인 구축 내용 대신 길게 나열하지 마세요.')


def literal_options(text):
    """Partition source text into contiguous excerpts without changing a byte.

    Prefer full lines and nearby whitespace boundaries. Every non-whitespace
    character remains covered; this is source segmentation, not extraction of
    facts or inference that an excerpt supports a generated claim.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('인용 선택지를 만들 원문이 없습니다.')
    options = []
    start = 0
    while start < len(text):
        end = start
        used = 0
        while end < len(text) and used + len(text[end].encode('utf-8')) <= MAX_QUOTE_BYTES:
            used += len(text[end].encode('utf-8'))
            end += 1
        if end < len(text):
            minimum = start + (end - start) // 2
            newline = text.rfind('\n', minimum, end)
            if newline >= minimum:
                end = newline + 1
            else:
                spaces = [i for i in range(minimum, end) if text[i].isspace()]
                if spaces:
                    end = spaces[-1] + 1
        quote = text[start:end].strip()
        if quote:
            options.append(quote)
        start = end
    return list(dict.fromkeys(options))


def literal_catalog(evidence):
    """Keep the authoritative chunk/document/quotation binding locally."""
    if not evidence:
        raise ValueError('인용 선택지에는 현재 입력 근거가 필요합니다.')
    catalog = {}
    count = 0
    for chunk in evidence:
        cid = chunk['chunk_id']
        if cid in catalog:
            raise ValueError('중복 청크의 인용 선택지를 만들 수 없습니다.')
        quotes = literal_options(chunk['text'])
        count += len(quotes)
        if count > MAX_OPTIONS:
            raise ValueError('원문 인용 선택지 수가 상한을 넘었습니다.')
        catalog[cid] = {'doc_id': chunk['doc_id'], 'quotes': quotes, 'text': chunk['text']}
    return catalog


def literal_schema(evidence):
    """Put short ASCII selectors in the wire schema, never source literals.

    Avoid repeated anyOf object branches. General JSON Schema validity does
    not by itself establish that the provider accepts a particular schema.
    """
    catalog = literal_catalog(evidence)
    schema = copy.deepcopy(ANSWER_SCHEMA)
    claim = schema['properties']['claims']['items']['properties']
    claim['doc_id'] = {'type': 'string', 'enum': list(dict.fromkeys(c['doc_id'] for c in evidence))}
    citation = claim['citations']['items']['properties']
    citation['chunk_id'] = {'type': 'string', 'enum': list(catalog)}
    citation['quote'] = {'type': 'string', 'enum': list(quote_references(evidence))}
    return schema


def quote_references(evidence):
    """One request-local ID per exact, chunk-owned source excerpt."""
    result = {}
    for cid, chunk in literal_catalog(evidence).items():
        for quote in chunk['quotes']:
            result[f'q{len(result) + 1:04d}'] = {
                'chunk_id': cid, 'doc_id': chunk['doc_id'], 'quote': quote}
    return result


def accepts_citations(payload, schema, evidence):
    """Enforce exact selections AND their actual chunk and document owners.

    The wire's two flat enums cannot express their correlation. Never infer
    a replacement chunk or repair a cross-paired choice; reject it instead.
    """
    properties = schema['properties']['claims']['items']['properties']
    docs = set(properties['doc_id']['enum'])
    choices = properties['citations']['items']['properties']
    allowed_ids = set(choices['chunk_id']['enum'])
    allowed_refs = set(choices['quote']['enum'])
    references = quote_references(evidence)
    try:
        return all(claim['doc_id'] in docs and all(
            cite['chunk_id'] in allowed_ids and cite['quote'] in allowed_refs
            and cite['quote'] in references
            and references[cite['quote']]['doc_id'] == claim['doc_id']
            and references[cite['quote']]['chunk_id'] == cite['chunk_id']
            for cite in claim['citations']) for claim in payload['claims'])
    except (KeyError, TypeError):
        return False


class LiteralQuoteClient:
    """An adapter around one retained guarded client; no SDK/client creation."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.prepared_requests = 0

    @property
    def config(self):
        return self.delegate.config

    def _request(self, task, question, evidence, schema=None):
        if schema is not None:
            raise ValueError('사업 범위 인용 계약에 다른 응답 스키마를 섞을 수 없습니다.')
        if task not in {'qa', COMPARISON_DOCUMENT_TASK}:
            raise ValueError('지원하지 않는 인용 계약 작업입니다.')
        annotated = copy.deepcopy(evidence)
        references = quote_references(evidence)
        for chunk in annotated:
            chunk['citation_options'] = [{'ref': ref, 'quote': choice['quote']}
                for ref, choice in references.items() if choice['chunk_id'] == chunk['chunk_id']]
        return {'task': QA_TASK if task == 'qa' else task,
            'question': question + '\n\n' + GUIDE, 'evidence': annotated,
            'schema': literal_schema(evidence)}

    def preflight_requests(self, requests):
        prepared = [self._request(**request) for request in requests]
        self.prepared_requests = len(prepared)
        preflight = getattr(self.delegate, 'preflight_requests', None)
        return preflight(prepared) if callable(preflight) else None

    def generate(self, task, question, evidence, *, schema=None):
        request = self._request(task, question, evidence, schema)
        result = self.delegate.generate(**request)
        if not accepts_citations(result.get('payload'), request['schema'], evidence):
            raise LLMError('Model citation is outside the literal selection contract', diagnostic={
                'code': 'invalid_answer_structure', 'request_fingerprint': result.get('request_fingerprint')})
        result = copy.deepcopy(result)
        references = quote_references(evidence)
        for claim in result['payload']['claims']:
            for cite in claim['citations']:
                cite['quote'] = references[cite['quote']]['quote']
        result['literal_citation_contract'] = {'version': VERSION,
            'quote_options': len(request['schema']['properties']['claims']['items']['properties']
                                 ['citations']['items']['properties']['quote']['enum']),
            'chunk_quote_binding_checked_locally': True,
            'reference_ids_resolved_locally': True,
            'selection_valid': True, 'source_correctness_still_requires_registry': True,
            'semantic_correctness_verified': False}
        return result
