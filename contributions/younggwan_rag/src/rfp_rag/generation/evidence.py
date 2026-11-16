"""Validate answer quotations against immutable source locations, not model belief.

Passing this validator establishes where a quotation came from. It does not
establish that the model's claim follows from that quotation, or that the
retrieved context was sufficient to answer the user's question.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import re


CATEGORIES = ('overview', 'budget', 'duration', 'eligibility', 'requirements',
              'submission', 'evaluation', 'deliverables', 'risks', 'other')
STATUSES = ('answered', 'partial', 'insufficient_evidence')
CATEGORY_LABELS = dict(zip(CATEGORIES, ('사업 개요', '예산', '사업기간', '참가자격',
    '요구사항', '제출', '평가기준', '산출물', '유의사항', '기타')))

ANSWER_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'status': {'type': 'string', 'enum': list(STATUSES)},
        'claims': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'doc_id': {'type': 'string'},
                'category': {'type': 'string', 'enum': list(CATEGORIES)},
                'text': {'type': 'string'},
                'citations': {'type': 'array', 'items': {
                    'type': 'object', 'additionalProperties': False,
                    'properties': {'chunk_id': {'type': 'string'}, 'quote': {'type': 'string'}},
                    'required': ['chunk_id', 'quote']}}},
            'required': ['doc_id', 'category', 'text', 'citations']}},
        'missing_information': {'type': 'array', 'items': {'type': 'string'}}},
    'required': ['status', 'claims', 'missing_information']}


class EvidenceError(ValueError):
    """A context or citation is not attributable to the registered source."""


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _indexable(block, collection):
    method = str(block.get('source_method', '')).lower()
    kind = str(block.get('kind', '')).lower()
    if block.get('reference_view') or 'ocr' in method or 'ocr' in kind or block.get('review_status') == 'unreviewed':
        return False
    if collection == 'supplemental_blocks' or method == 'reviewed_visual_transcription' or kind == 'verified_transcription':
        return method == 'reviewed_visual_transcription' and block.get('review_status') == 'visual_verified'
    return True


def _boundary(block):
    if block.get('page') is not None:
        return 'page', block['page']
    section = block.get('section')
    if section is None:
        match = re.match(r'(BodyText/Section\d+|Contents/section\d+\.xml)', str(block.get('locator', '')))
        section = match.group(1) if match else None
    return ('section', section) if section is not None else ('unknown_block', block['block_id'])


def _merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def _nonblank_string(value):
    return isinstance(value, str) and bool(value.strip())


class EvidenceRegistry:
    """Trusted extracted documents and canonical chunks for one data version.

    Documents/chunks are treated as read-only input. Chunk snapshots are hashed
    on registration and checked before use, so mutating a registered chunk is
    rejected rather than silently changing a citation's content.
    """

    def __init__(self, documents, chunks):
        self.documents_by_id = {}
        self.aliases = {}
        self.source_hashes = {}
        self._blocks = {}
        self._chunks = {}
        self._chunk_hashes = {}
        for document in documents:
            if not isinstance(document, dict):
                raise EvidenceError('문서 자료형이 올바르지 않습니다.')
            did = document.get('doc_id')
            if not _nonblank_string(did) or did in self.documents_by_id:
                raise EvidenceError('문서 ID가 누락되었거나 중복됐습니다.')
            self.documents_by_id[did] = document
            self.aliases[did] = document.get('canonical_doc_id', did)
        for did, document in self.documents_by_id.items():
            canonical = self.aliases[did]
            if not _nonblank_string(canonical):
                raise EvidenceError('대표 문서 ID가 올바르지 않습니다.')
            original = self.documents_by_id.get(canonical)
            if original is None or self.aliases.get(canonical) != canonical:
                raise EvidenceError('대표 문서 ID 연결이 올바르지 않습니다.')
            sha = document.get('sha256')
            if not isinstance(sha, str) or not re.fullmatch('[a-fA-F0-9]{64}', sha) or sha != original.get('sha256'):
                raise EvidenceError('문서 버전 또는 사본의 원본 SHA-256이 다릅니다.')
            if did != canonical:
                continue
            if document.get('is_duplicate_alias'):
                raise EvidenceError('대표 문서가 중복 사본으로 표시됐습니다.')
            self.source_hashes[canonical] = sha
            for collection in ('blocks', 'supplemental_blocks'):
                for block in document.get(collection, []):
                    if not _indexable(block, collection):
                        continue
                    key = canonical, collection, block.get('block_id')
                    if not _nonblank_string(key[2]) or key in self._blocks or not isinstance(block.get('text'), str):
                        raise EvidenceError('원문 블록 ID 또는 본문이 올바르지 않습니다.')
                    self._blocks[key] = block
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise EvidenceError('청크 자료형이 올바르지 않습니다.')
            cid = chunk.get('chunk_id')
            if not _nonblank_string(cid) or cid in self._chunks:
                raise EvidenceError('청크 ID가 누락되었거나 중복됐습니다.')
            self._chunks[cid] = chunk
            try:
                self._chunk_hashes[cid] = _fingerprint(chunk)
            except (ValueError, TypeError) as error:
                raise EvidenceError('청크를 안정적인 JSON으로 직렬화할 수 없습니다.') from error

    @property
    def canonical_doc_ids(self):
        return sorted(self.source_hashes)

    @property
    def chunk_ids(self):
        return list(self._chunks)

    def resolve_doc_ids(self, ids):
        if ids is None:
            return self.canonical_doc_ids
        try:
            if isinstance(ids, dict):
                raise TypeError
            requested = [ids] if isinstance(ids, str) else list(ids)
        except TypeError as error:
            raise EvidenceError('문서 범위는 ID 문자열 또는 ID 목록이어야 합니다.') from error
        if any(not _nonblank_string(value) or value not in self.aliases for value in requested):
            raise EvidenceError('선택한 문서 ID가 원문 목록에 없습니다. 검색 범위를 자동 확대하지 않습니다.')
        return list(dict.fromkeys(self.aliases[value] for value in requested))

    def get_chunk(self, chunk_id):
        if not _nonblank_string(chunk_id) or chunk_id not in self._chunks:
            raise EvidenceError('청크 ID가 원문 레지스트리에 없습니다.')
        return self.validate_chunks([self._chunks[chunk_id]])[0]

    def _check_registered_chunk(self, chunk):
        cid = chunk['chunk_id']
        if _fingerprint(chunk) != self._chunk_hashes[cid]:
            raise EvidenceError('등록한 뒤 청크 내용이 바뀌었습니다.')
        # Raw source slices are checked on each call, including previously used
        # chunks, because callers may mutate their document objects in Python.
        did = chunk.get('doc_id')
        if did not in self.source_hashes or chunk.get('canonical_doc_id', did) != did:
            raise EvidenceError('대표 원문 문서에 속하지 않는 청크입니다.')
        if chunk.get('sha256') != self.source_hashes[did]:
            raise EvidenceError('청크의 원본 SHA-256이 다릅니다.')
        document = self.documents_by_id[did]
        if document.get('sha256') != self.source_hashes[did]:
            raise EvidenceError('등록한 뒤 문서 버전이 바뀌었습니다.')
        if 'file_name' in chunk and chunk['file_name'] != document.get('file_name'):
            raise EvidenceError('청크의 원본 파일명이 다릅니다.')
        text = chunk.get('text')
        if not _nonblank_string(text):
            raise EvidenceError('청크 본문이 비어 있습니다.')
        if chunk.get('length_unit') != 'utf8_bytes' or chunk.get('text_bytes') != len(text.encode('utf-8')):
            raise EvidenceError('청크 본문의 byte 길이 또는 단위가 다릅니다.')
        spans = chunk.get('source_spans')
        if not isinstance(spans, list) or not spans:
            raise EvidenceError('청크에 원문 위치 범위가 없습니다.')
        covered, boundaries, collections, provenances = [], set(), set(), set()
        pages, sections = set(), set()
        for span in spans:
            if not isinstance(span, dict):
                raise EvidenceError('원문 위치 범위의 형식이 올바르지 않습니다.')
            collection = span.get('collection')
            if collection not in ('blocks', 'supplemental_blocks'):
                raise EvidenceError('중복 reference view 또는 미검수 보완 자료는 근거로 쓸 수 없습니다.')
            block = self._blocks.get((did, collection, span.get('block_id')))
            if block is None or not _indexable(block, collection):
                raise EvidenceError('원문 블록이 없거나 OCR/미검수 자료입니다.')
            a, b, x, y = (span.get(key) for key in ('char_start', 'char_end', 'chunk_start', 'chunk_end'))
            if not all(type(value) is int for value in (a, b, x, y)) or not (0 <= a < b <= len(block['text']) and 0 <= x < y <= len(text)):
                raise EvidenceError('원문 또는 청크의 문자 범위가 올바르지 않습니다.')
            if b-a != y-x or block['text'][a:b] != text[x:y]:
                raise EvidenceError('청크 내용이 연결된 원문과 다릅니다.')
            if not _nonblank_string(block.get('locator')) or span.get('locator') != block['locator']:
                raise EvidenceError('원문 위치 표시가 다릅니다.')
            for field in ('page', 'section'):
                if span.get(field) != block.get(field):
                    raise EvidenceError('원문 페이지 또는 섹션 메타데이터가 다릅니다.')
            if block.get('page') is not None:
                pages.add(block['page'])
            if block.get('section') is not None:
                sections.add(block['section'])
            covered.append((x, y)); boundaries.add(_boundary(block)); collections.add(collection)
            provenances.add('reviewed' if block.get('source_method') == 'reviewed_visual_transcription' else 'automatic')
        if len(boundaries) != 1 or len(collections) != 1 or len(provenances) != 1:
            raise EvidenceError('청크가 다른 원문 경계 또는 검수 방식을 혼합했습니다.')
        for field, values in (('page', pages), ('section', sections)):
            if field in chunk and (len(values) != 1 or chunk[field] not in values):
                raise EvidenceError('청크의 페이지 또는 섹션 표시가 다릅니다.')
        if 'reviewed' in provenances:
            if chunk.get('source_method') != 'reviewed_visual_transcription' or chunk.get('kind') != 'verified_transcription' or chunk.get('review_status') != 'visual_verified':
                raise EvidenceError('검수 전사의 출처 또는 검수 상태 표시가 다릅니다.')
        elif chunk.get('source_method') != 'raw_parser' or chunk.get('kind') == 'verified_transcription':
            raise EvidenceError('자동 추출 원문의 출처 표시가 다릅니다.')
        last = 0
        for a, b in _merged(covered):
            if text[last:a].strip():
                raise EvidenceError('원문으로 설명되지 않은 청크 문자가 있습니다.')
            last = b
        if text[last:].strip():
            raise EvidenceError('원문으로 설명되지 않은 청크 끝 문자가 있습니다.')

    def validate_chunks(self, chunks, allowed_doc_ids=None):
        scope = None if allowed_doc_ids is None else set(self.resolve_doc_ids(allowed_doc_ids))
        result, seen = [], set()
        for supplied in chunks:
            if not isinstance(supplied, dict):
                raise EvidenceError('검색 문맥 청크의 형식이 올바르지 않습니다.')
            cid = supplied.get('chunk_id')
            if not _nonblank_string(cid) or cid not in self._chunks:
                raise EvidenceError('검색 문맥에 등록되지 않은 청크가 있습니다.')
            canonical = self._chunks[cid]
            # Retrieval rank/score additions are allowed, but every registered
            # source field must still be present and exactly match its snapshot.
            if any(key not in supplied or supplied[key] != value for key, value in canonical.items()):
                raise EvidenceError('검색 청크의 본문 또는 출처 메타데이터가 원본과 다릅니다.')
            if scope is not None and canonical.get('doc_id') not in scope:
                raise EvidenceError('선택한 문서 범위를 벗어난 청크입니다.')
            try:
                self._check_registered_chunk(canonical)
            except (TypeError, KeyError, ValueError) as error:
                if isinstance(error, EvidenceError):
                    raise
                raise EvidenceError('청크 또는 원문 위치의 자료형이 올바르지 않습니다.') from error
            if cid not in seen:
                seen.add(cid); result.append(canonical)
        return result

    def _resolve_citation(self, citation, context_by_id, claim_doc_id):
        if not isinstance(citation, dict) or set(citation) != {'chunk_id', 'quote'}:
            raise EvidenceError('인용은 chunk_id와 quote 두 필드만 가져야 합니다.')
        cid, quote = citation['chunk_id'], citation['quote']
        if not _nonblank_string(cid) or cid not in context_by_id:
            raise EvidenceError('모델이 제공받지 않은 청크를 인용했습니다.')
        if not _nonblank_string(quote):
            raise EvidenceError('인용문이 비어 있습니다.')
        chunk = context_by_id[cid]
        if chunk['doc_id'] != claim_doc_id:
            raise EvidenceError('주장과 인용의 문서가 다릅니다.')
        start = chunk['text'].find(quote)
        if start < 0:
            raise EvidenceError('인용문이 청크 원문과 정확히 일치하지 않습니다.')
        end = start + len(quote)
        ranges, coverage = [], []
        for span in chunk['source_spans']:
            left, right = max(start, span['chunk_start']), min(end, span['chunk_end'])
            if left >= right:
                continue
            item = {'collection': span['collection'], 'block_id': span['block_id'],
                    'locator': span['locator'],
                    'char_start': span['char_start'] + left - span['chunk_start'],
                    'char_end': span['char_start'] + right - span['chunk_start'],
                    'chunk_start': left, 'chunk_end': right}
            for field in ('page', 'section', 'table_id', 'row', 'col'):
                if field in span:
                    item[field] = span[field]
            ranges.append(item); coverage.append((left-start, right-start))
        last = 0
        for a, b in _merged(coverage):
            if quote[last:a].strip():
                raise EvidenceError('인용문의 일부가 실제 원문 범위에 속하지 않습니다.')
            last = b
        if quote[last:].strip() or not ranges:
            raise EvidenceError('인용문을 실제 원문 위치로 복원할 수 없습니다.')
        result = {'chunk_id': cid, 'quote': quote, 'quote_start': start, 'quote_end': end,
                  'doc_id': chunk['doc_id'], 'sha256': chunk['sha256'],
                  'file_name': self.documents_by_id[chunk['doc_id']].get('file_name', chunk['doc_id']),
                  'locator': chunk.get('locator'), 'source_ranges': ranges,
                  'source_method': chunk['source_method'],
                  'quality_flags': copy.deepcopy(chunk.get('quality_flags', [])),
                  'provenance_verified': True, 'semantic_entailment_verified': False}
        for field in ('page', 'section', 'review_status'):
            if field in chunk:
                result[field] = chunk[field]
        return result

    def validate_answer(self, payload, context_chunks, allowed_doc_ids=None):
        errors, claims, missing = [], [], []
        status = 'insufficient_evidence'
        try:
            context = self.validate_chunks(context_chunks, allowed_doc_ids)
            context_by_id = {chunk['chunk_id']: chunk for chunk in context}
            scope = None if allowed_doc_ids is None else set(self.resolve_doc_ids(allowed_doc_ids))
            if not isinstance(payload, dict) or set(payload) != {'status', 'claims', 'missing_information'}:
                raise EvidenceError('응답의 최상위 필드가 정해진 스키마와 다릅니다.')
            status = payload['status']
            if status not in STATUSES or not isinstance(status, str):
                raise EvidenceError('응답 상태 값이 올바르지 않습니다.')
            raw_claims, raw_missing = payload['claims'], payload['missing_information']
            if not isinstance(raw_claims, list) or len(raw_claims) > 1000:
                raise EvidenceError('주장 목록의 형식 또는 개수가 올바르지 않습니다.')
            if not isinstance(raw_missing, list) or len(raw_missing) > 1000 or any(not _nonblank_string(value) for value in raw_missing):
                raise EvidenceError('미확인 항목 목록의 형식이 올바르지 않습니다.')
            if (status in ('answered', 'partial') and not raw_claims) or (status == 'insufficient_evidence' and raw_claims):
                raise EvidenceError('응답 상태와 근거가 있는 주장 수가 서로 맞지 않습니다.')
            if status == 'answered' and raw_missing:
                raise EvidenceError('완료 상태인데 미확인 항목이 남아 있습니다.')
            if status == 'partial' and not raw_missing:
                raise EvidenceError('부분 답변의 미확인 항목이 표시되지 않았습니다.')
            for number, claim in enumerate(raw_claims, 1):
                if not isinstance(claim, dict) or set(claim) != {'doc_id', 'category', 'text', 'citations'}:
                    raise EvidenceError(f'주장 {number}의 필드가 정해진 스키마와 다릅니다.')
                did = claim['doc_id']
                if not _nonblank_string(did) or did not in self.source_hashes:
                    raise EvidenceError(f'주장 {number}의 대표 문서 ID가 올바르지 않습니다.')
                if scope is not None and did not in scope:
                    raise EvidenceError(f'주장 {number}가 사용자 선택 문서 범위를 벗어났습니다.')
                if claim['category'] not in CATEGORIES or not _nonblank_string(claim['text']):
                    raise EvidenceError(f'주장 {number}의 분류 또는 내용이 올바르지 않습니다.')
                citations = claim['citations']
                if not isinstance(citations, list) or not citations or len(citations) > 100:
                    raise EvidenceError(f'주장 {number}의 인용이 없거나 형식이 올바르지 않습니다.')
                resolved = [self._resolve_citation(citation, context_by_id, did) for citation in citations]
                claims.append({'doc_id': did, 'category': claim['category'], 'text': claim['text'], 'citations': resolved})
            missing = list(raw_missing)
        except (EvidenceError, TypeError, ValueError, KeyError) as error:
            errors.append(str(error) if isinstance(error, EvidenceError) else '응답 또는 원문 범위의 자료형이 올바르지 않습니다.')
            # Fail the whole response closed; a later invalid claim must not
            # leak earlier model text through a caller's generic renderer.
            claims, missing, status = [], [], 'insufficient_evidence'
        return {'valid': not errors, 'status': status, 'claims': claims,
                'missing_information': missing, 'errors': errors,
                'provenance_verified': not errors,
                'semantic_entailment_verified': False,
                'answer_correctness_verified': False,
                'validation_scope': 'Exact quotation provenance and document scope only; not semantic entailment or whole-document completeness.'}

    @staticmethod
    def render(result):
        if not isinstance(result, dict) or not result.get('valid'):
            return '인용 근거 검증을 통과하지 못해 생성 답변을 표시하지 않았습니다.'
        def safe(value):
            text = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            for character in ('\\', '`', '*', '_', '[', ']'):
                text = text.replace(character, '\\' + character)
            return text
        labels = {'answered': '근거를 연결한 답변', 'partial': '확보한 근거 범위의 부분 답변',
                  'insufficient_evidence': '제공된 검색 근거만으로 답변을 확정하기 어렵습니다.'}
        lines = [labels.get(result.get('status'), '근거 확인 필요'), '']
        for claim in result.get('claims', []):
            lines.append('- **' + CATEGORY_LABELS.get(claim['category'], '기타') + '**: ' + safe(claim['text']))
            for citation in claim['citations']:
                lines.append('  - 출처: ' + safe(citation['file_name']) + ' · ' + safe(citation.get('locator') or '위치 미상'))
                lines.append('    원문: “' + safe(citation['quote']).replace('\n', ' / ') + '”')
        if result.get('missing_information'):
            lines.extend(['', '추가로 확인할 항목:'])
            lines.extend('- ' + safe(item) for item in result['missing_information'])
        lines.extend(['', '인용문과 원문 위치를 대조했습니다. 인용문이 주장 전체를 뒷받침하는지는 별도 검토가 필요합니다.'])
        return '\n'.join(lines)
