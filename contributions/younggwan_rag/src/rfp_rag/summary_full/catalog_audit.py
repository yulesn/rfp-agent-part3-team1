"""Offline document-ID selection and corpus-wide identity/source checks.

All original IDs remain selectable. Duplicate sources map to one canonical
document and one ordered set of chunks. Nothing here contacts a model service,
changes the index, or judges the semantic quality of an answer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import time


class CatalogError(ValueError):
    """Catalog, source snapshot, or selected document IDs disagree."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _chunks_fingerprint(chunks):
    h = hashlib.sha256()
    for chunk in chunks:
        raw = json.dumps(chunk, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
        h.update(len(raw).to_bytes(8, 'big'))
        h.update(raw)
    return h.hexdigest()


def _catalog_maps(env):
    """Validate source/catalog identity without reading eval questions or APIs."""
    try:
        documents = list(env['documents'])
        chunks = list(env['chunks'])
        index, registry = env['index'], env['registry']
        metadata = index.get_metadata()
        catalog = index.get_catalog()
    except (KeyError, TypeError, AttributeError) as error:
        raise CatalogError('원문·청크·인덱스·레지스트리가 모두 로드된 환경이 필요합니다.') from error
    if not documents or not catalog:
        raise CatalogError('조회할 문서 목록이 비어 있습니다.')
    originals = {}
    for doc in documents:
        did = doc.get('doc_id')
        if not isinstance(did, str) or not did.strip() or did in originals:
            raise CatalogError('원본 문서 ID가 비어 있거나 중복되었습니다.')
        originals[did] = doc
    canonical = {}
    for row in catalog:
        did = row.get('doc_id')
        if not isinstance(did, str) or not did.strip() or did in canonical:
            raise CatalogError('대표 문서 ID가 비어 있거나 중복되었습니다.')
        if row.get('canonical_doc_id', did) != did:
            raise CatalogError('인덱스 목록의 대표 문서 ID가 일치하지 않습니다.')
        canonical[did] = row
    aliases = {did: doc.get('canonical_doc_id', did) for did, doc in originals.items()}
    if set(aliases.values()) != set(canonical):
        raise CatalogError('원문 대표 문서와 전체 검색 대상 목록이 다릅니다.')
    if metadata.get('alias_map') != aliases:
        raise CatalogError('인덱스와 원문 목록의 원본·대표 문서 ID 연결이 다릅니다.')
    if set(registry.canonical_doc_ids) != set(canonical):
        raise CatalogError('원문 레지스트리와 인덱스의 대표 문서 목록이 다릅니다.')
    if metadata.get('canonical_document_count') != len(canonical):
        raise CatalogError('인덱스의 대표 문서 개수 기록이 다릅니다.')
    if len({row.get('sha256') for row in catalog}) != len(catalog):
        raise CatalogError('동일한 원본 SHA가 여러 대표 문서로 중복 인덱싱되었습니다.')
    grouped = defaultdict(set)
    for did, target in aliases.items():
        doc = originals[did]
        original = originals.get(target)
        row = canonical.get(target)
        if original is None or row is None or aliases.get(target) != target:
            raise CatalogError('원본 문서 ID가 조회 가능한 대표 문서로 연결되지 않습니다.')
        sha = doc.get('sha256')
        if sha != original.get('sha256') or sha != row.get('sha256'):
            raise CatalogError('원문·사본·인덱스의 SHA 버전이 다릅니다.')
        if registry.source_hashes.get(target) != sha:
            raise CatalogError('원문 레지스트리의 SHA 버전이 다릅니다.')
        if registry.aliases.get(did) != target:
            raise CatalogError('원문 레지스트리의 원본·대표 ID 연결이 다릅니다.')
        if bool(doc.get('is_duplicate_alias', did != target)) != (did != target):
            raise CatalogError('중복 사본 표시가 원본·대표 ID 연결과 다릅니다.')
        grouped[target].add(did)
    for did, row in canonical.items():
        if set(row.get('alias_doc_ids', [])) != grouped[did]:
            raise CatalogError('대표 문서에 연결된 원본 ID 목록이 다릅니다.')
    return originals, canonical, aliases, metadata, chunks


def build_catalog(env):
    """Return every original ID, its canonical ID, source version and chunk count.

    Original IDs are not truncated or replaced with model-generated IDs. The
    output has one row per original file; aliases explicitly identify copies.
    """
    originals, canonical, aliases, _, chunks = _catalog_maps(env)
    counts = Counter(c.get('doc_id') for c in chunks)
    return [{'doc_id': did, 'original_doc_id': did,
             'canonical_doc_id': aliases[did], 'is_alias': did != aliases[did],
             'file_name': doc.get('file_name', did),
             'canonical_file_name': canonical[aliases[did]].get('file_name', aliases[did]),
             'sha256': doc['sha256'], 'chunk_count': counts[aliases[did]]}
            for did, doc in originals.items()]


def select_documents(env, requested='all'):
    """Resolve any source/canonical ID and deduplicate without widening scope.

    Only the explicit string ``all`` selects the entire canonical corpus.
    Unknown IDs, None, empty selection, dict/set, or non-string members raise
    ValueError. A list preserves caller order while aliases deduplicate.
    """
    originals, canonical, aliases, _, _ = _catalog_maps(env)
    if requested == 'all':
        return list(canonical)
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, (list, tuple)) or not requested:
        raise CatalogError('문서 ID 또는 ID 목록을 지정하세요. 전체 선택은 명시적으로 all을 사용하세요.')
    unknown = [did for did in requested
               if not isinstance(did, str) or not did.strip() or did not in originals]
    if unknown:
        raise CatalogError('목록에 없는 문서 ID입니다. 검색 범위를 자동으로 바꾸지 않습니다: '
                           + ', '.join(str(did) for did in unknown[:5]))
    selected = list(dict.fromkeys(aliases[did] for did in requested))
    resolved = env['index'].resolve_doc_ids(list(requested))
    if resolved != selected or env['registry'].resolve_doc_ids(list(requested)) != selected:
        raise CatalogError('문서 선택 결과가 원문·인덱스의 ID 연결과 다릅니다.')
    return selected


def get_validated_document_chunks(env, doc_id):
    """Return every registered chunk for one original/canonical document ID."""
    canonical = select_documents(env, [doc_id])[0]
    chunks = env['index'].get_document_chunks(doc_id)
    expected = [c['chunk_id'] for c in env['chunks'] if c['doc_id'] == canonical]
    raw_ids = [c.get('chunk_id') for c in chunks]
    if raw_ids != expected or len(raw_ids) != len(set(raw_ids)):
        raise CatalogError('인덱스가 반환한 문서 청크의 순서·개수·중복이 다릅니다.')
    checked = env['registry'].validate_chunks(chunks, allowed_doc_ids=[canonical])
    actual = [c['chunk_id'] for c in checked]
    if not expected or actual != expected or len(actual) != len(set(actual)):
        raise CatalogError('문서의 전체 청크가 누락·중복되었거나 원문 순서가 다릅니다.')
    return list(checked)


def audit_catalog(env):
    """Check every available source ID and every canonical chunk, without API.

    Failing cases remain failures. The report contains IDs/counts/fingerprints,
    not document bodies, generated answers, or semantic accuracy estimates.
    """
    started = time.perf_counter()
    checks, errors, per_document = {}, [], []
    try:
        originals, canonical, aliases, metadata, chunks = _catalog_maps(env)
    except (ValueError, TypeError, KeyError) as error:
        return {'version': 'phase5-all-document-id-audit-v1', 'passed': False,
            'errors': [str(error)], 'checks': {}, 'api_calls': 0,
            'semantic_quality_verified': False, 'generated_answer_quality_verified': False,
            'elapsed_seconds': round(time.perf_counter() - started, 3)}
    expected_by_doc = defaultdict(list)
    for c in chunks:
        expected_by_doc[c.get('doc_id')].append(c)
    all_ids = [c.get('chunk_id') for c in chunks]
    checks['all_original_ids_have_one_canonical_target'] = set(aliases.values()) == set(canonical)
    checks['canonical_sources_indexed_once'] = len({r['sha256'] for r in canonical.values()}) == len(canonical)
    checks['global_chunk_ids_unique'] = len(all_ids) == len(set(all_ids))
    checks['chunk_count_matches_metadata'] = metadata.get('chunk_count') == len(chunks)
    checks['registered_chunk_ids_match_saved_order'] = list(env['registry'].chunk_ids) == all_ids
    checks['chunk_document_scope_matches_catalog'] = set(expected_by_doc) == set(canonical)
    checks['saved_chunks_fingerprint_matches_index'] = _chunks_fingerprint(chunks) == metadata.get('chunks_fingerprint')
    aliases_resolved = 0
    for did, target in aliases.items():
        try:
            if env['index'].resolve_doc_ids([did]) != [target] or env['registry'].resolve_doc_ids([did]) != [target]:
                raise CatalogError('원본 문서 ID의 대표 연결이 다릅니다.')
            if did != target:
                alias_chunks = env['index'].get_document_chunks(did)
                if [c['chunk_id'] for c in alias_chunks] != [c['chunk_id'] for c in expected_by_doc[target]]:
                    raise CatalogError('중복 사본 ID로 조회한 청크가 대표 문서와 다릅니다.')
            aliases_resolved += 1
        except (ValueError, KeyError, TypeError) as error:
            errors.append({'doc_id': did, 'code': 'original_id_roundtrip_failed', 'message': str(error)})
    checks['every_original_id_resolves'] = aliases_resolved == len(originals)
    seen = []
    for did, row in canonical.items():
        item = {'doc_id': did, 'file_name': row.get('file_name', did), 'sha256': row['sha256'],
                'chunk_count': len(expected_by_doc[did]), 'passed': False}
        try:
            selected = env['index'].get_document_chunks(did)
            raw_ids = [c.get('chunk_id') for c in selected]
            expected_ids = [c['chunk_id'] for c in expected_by_doc[did]]
            if raw_ids != expected_ids or len(raw_ids) != len(set(raw_ids)):
                raise CatalogError('인덱스 원본 청크의 순서·개수·중복이 다릅니다.')
            validated = env['registry'].validate_chunks(selected, allowed_doc_ids=[did])
            ids = [c['chunk_id'] for c in validated]
            if not expected_ids or ids != expected_ids or len(ids) != len(set(ids)):
                raise CatalogError('전체 청크의 개수·순서·중복 검사가 통과하지 못했습니다.')
            if any(c['doc_id'] != did or c['sha256'] != row['sha256'] for c in validated):
                raise CatalogError('문서의 청크 ID 또는 SHA 버전이 다릅니다.')
            seen.extend(ids)
            item.update(passed=True, ordered_chunk_ids_fingerprint=_digest(ids),
                        first_chunk_id=ids[0], last_chunk_id=ids[-1])
        except (ValueError, KeyError, TypeError) as error:
            item['error'] = str(error)
            errors.append({'doc_id': did, 'code': 'canonical_source_chunk_audit_failed', 'message': str(error)})
        per_document.append(item)
    checks['every_canonical_document_chunks_verified'] = all(d['passed'] for d in per_document)
    checks['every_saved_chunk_seen_once'] = len(seen) == len(all_ids) and set(seen) == set(all_ids)
    sentinel = '__phase5_unknown_document_id__'
    while sentinel in originals:
        sentinel += '_'
    rejected = []
    for name, resolve in [('index', env['index'].resolve_doc_ids), ('registry', env['registry'].resolve_doc_ids)]:
        try:
            resolve([sentinel])
            rejected.append(False)
        except ValueError:
            rejected.append(True)
    checks['unknown_ids_raise_instead_of_widening_scope'] = all(rejected)
    errors.extend({'code': key} for key, passed in checks.items() if not passed)
    return {'version': 'phase5-all-document-id-audit-v1', 'passed': not errors and all(checks.values()),
            'original_document_count': len(originals), 'canonical_document_count': len(canonical),
            'duplicate_alias_count': len(originals)-len(canonical), 'chunk_count': len(chunks),
            'original_ids_resolved': aliases_resolved, 'canonical_documents_verified': sum(d['passed'] for d in per_document),
            'index_fingerprint': metadata.get('index_fingerprint'), 'checks': checks, 'errors': errors,
            'documents': per_document, 'api_calls': 0, 'semantic_quality_verified': False,
            'generated_answer_quality_verified': False,
            'elapsed_seconds': round(time.perf_counter()-started, 3),
            'scope': '전체 원본 ID·대표 ID·중복 연결·원문 SHA·청크 내용과 순서의 오프라인 검사. 실제 생성 답변의 정확성 평가는 별도입니다.'}
