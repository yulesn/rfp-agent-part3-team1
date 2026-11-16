"""Recover source quotations, then run the original strict validator.

Every non-whitespace character, the cited chunk, document, SHA and raw offsets
remain authoritative. This never paraphrases, changes digits/punctuation,
searches other documents or invokes a model. Ambiguous recovery is rejected.
Up to eight uniquely matched whole-line source fragments are represented as
separate exact citations, with all omitted gaps recorded for human review.
A false continuous quote is never kept or presented as semantic verification.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3


RECOVERY_POLICY = 'bounded_lines_and_literal_reference_rebuild_v4'
MIN_SPLIT_FRAGMENT_CHARS = 12
MAX_SPLIT_FRAGMENTS = 8
MAX_SPLIT_QUOTE_LINES = 256
MAX_SPLIT_SOURCE_LINES = 2048


def _compact_with_positions(text):
    positions = [i for i, char in enumerate(text) if not char.isspace()]
    return ''.join(text[i] for i in positions), positions


def _recover_whitespace(source, quote):
    needle, _ = _compact_with_positions(quote)
    haystack, positions = _compact_with_positions(source)
    if not needle:
        return None, 'blank_quote'
    start = haystack.find(needle)
    if start < 0:
        return None, 'non_whitespace_characters_differ'
    if haystack.find(needle, start + 1) >= 0:
        return None, 'ambiguous_whitespace_match'
    left, right = positions[start], positions[start + len(needle) - 1] + 1
    restored = source[left:right]
    assert ''.join(restored.split()) == ''.join(quote.split())
    return {'quote': restored, 'quote_start': left, 'quote_end': right}, 'whitespace_only'


def _source_lines(text):
    """Nonempty full lines with exact original character offsets."""
    result, offset = [], 0
    for raw in text.splitlines(keepends=True):
        positions = [i for i, char in enumerate(raw) if not char.isspace()]
        if positions:
            result.append({'key': ''.join(raw[i] for i in positions),
                           'start': offset + positions[0],
                           'end': offset + positions[-1] + 1})
        offset += len(raw)
    return result


def _recover_two_fragments(source, quote):
    """Unique ordered full-line blocks, exactly one nonblank omitted source gap.

    No character-level subsequence matching, intra-line omission, new words,
    cross-chunk search, reordered blocks or short isolated labels are accepted.
    """
    requested = [line['key'] for line in _source_lines(quote)]
    lines = _source_lines(source)
    if len(requested) < 2:
        return None, 'not_a_multiline_whole_line_quote'
    keys = [line['key'] for line in lines]
    candidates = {}
    for split in range(1, len(requested)):
        first, second = requested[:split], requested[split:]
        if min(len(''.join(first)), len(''.join(second))) < MIN_SPLIT_FRAGMENT_CHARS:
            continue
        starts_a = [i for i in range(len(keys) - len(first) + 1) if keys[i:i+len(first)] == first]
        starts_b = [i for i in range(len(keys) - len(second) + 1) if keys[i:i+len(second)] == second]
        for a in starts_a:
            for b in starts_b:
                if b <= a + len(first):  # Require at least one nonempty omitted source line.
                    continue
                x, y = lines[a]['start'], lines[a+len(first)-1]['end']
                z, w = lines[b]['start'], lines[b+len(second)-1]['end']
                identity = x, y, z, w
                candidates[identity] = [
                    {'quote': source[x:y], 'quote_start': x, 'quote_end': y},
                    {'quote': source[z:w], 'quote_start': z, 'quote_end': w}]
                if len(candidates) > 1:
                    return None, 'ambiguous_fragment_mapping'
    if not candidates:
        return None, 'no_unique_two_fragment_match'
    fragments = next(iter(candidates.values()))
    assert ''.join(''.join(f['quote'].split()) for f in fragments) == ''.join(quote.split())
    return fragments, 'noncontiguous_quote_split'


def _recover_line_fragments(source, quote):
    """Unique ordered full-line blocks within one already selected chunk.

    Every requested line must match a complete source line. No intra-line
    deletion, changed punctuation/digits, reordered text, cross-chunk search,
    or ambiguous placements are accepted. Consecutive source lines form one
    block; each block must contain at least twelve non-whitespace characters.
    """
    requested = [line['key'] for line in _source_lines(quote)]
    lines = _source_lines(source)
    if len(requested) < 2:
        return None, 'not_a_multiline_whole_line_quote'
    if len(requested) > MAX_SPLIT_QUOTE_LINES or len(lines) > MAX_SPLIT_SOURCE_LINES:
        return None, 'fragment_search_limit_exceeded'
    keys = [line['key'] for line in lines]
    positions = {}
    for i, key in enumerate(keys):
        positions.setdefault(key, []).append(i)

    @lru_cache(maxsize=20000)
    def solve(qpos, minimum, remaining):
        if qpos == len(requested):
            return ((),)
        if remaining == 0:
            return ()
        found = []
        for start in positions.get(requested[qpos], ()):
            if start < minimum:
                continue
            length = 0
            while (qpos + length < len(requested) and start + length < len(keys)
                   and requested[qpos + length] == keys[start + length]):
                length += 1
            # Longer blocks first; all valid alternatives are still inspected.
            for count in range(length, 0, -1):
                if sum(len(key) for key in requested[qpos:qpos+count]) < MIN_SPLIT_FRAGMENT_CHARS:
                    continue
                end = start + count - 1
                for tail in solve(qpos + count, end + 2, remaining - 1):
                    candidate = ((start, end),) + tail
                    if candidate not in found:
                        found.append(candidate)
                    if len(found) > 1:
                        return tuple(found)
        return tuple(found)

    matches = solve(0, 0, MAX_SPLIT_FRAGMENTS)
    if len(matches) > 1:
        return None, 'ambiguous_fragment_mapping'
    if not matches or len(matches[0]) < 2:
        return None, 'no_unique_bounded_fragment_match'
    fragments = [{'quote': source[lines[a]['start']:lines[b]['end']],
                  'quote_start': lines[a]['start'], 'quote_end': lines[b]['end']}
                 for a, b in matches[0]]
    assert ''.join(''.join(f['quote'].split()) for f in fragments) == ''.join(quote.split())
    return fragments, 'noncontiguous_quote_split'



def _literal_reference_candidates(context, claim, citation):
    cid, quote = citation.get('chunk_id'), citation.get('quote')

    if set(citation) != {'chunk_id', 'quote'} or not isinstance(cid, str):
        return []

    # 정상 형식의 ID를 임의로 다른 청크에 연결하지 않습니다.
    if re.fullmatch(r'rfp:[0-9a-f]{16}:[0-9a-f]{24}', cid):
        return []

    chunks = [
        c for c in context.values()
        if c['doc_id'] == claim.get('doc_id')
    ]
    hashes = {c['sha256'] for c in chunks}

    if len(hashes) != 1:
        return []
    if not cid.startswith('rfp:' + next(iter(hashes))[:12]):
        return []
    if not isinstance(quote, str) or not quote.strip():
        return []

    # 현재 배치·같은 문서에서 원문이 정확히 일치하는 경우만 사용합니다.
    matches = [
        c['chunk_id'] for c in chunks
        if quote in c['text']
    ]

    # 반복되는 구절은 하나를 추측하지 않고 모든 일치 청크를 표시합니다.
    return matches if 1 <= len(matches) <= 8 else []


def validate_with_quote_recovery(registry, payload, context_chunks, allowed_doc_ids=None):
    """A formatting recovery layer; EvidenceRegistry itself remains strict.

    Diagnostic quotations are unvalidated model data, never normal answer text.
    Cache payloads and caller objects are not mutated. A single unresolved error
    still prevents the whole model answer from being rendered.
    """
    strict = registry.validate_answer(payload, context_chunks, allowed_doc_ids=allowed_doc_ids)
    report = {'policy': RECOVERY_POLICY, 'corrected_citations': 0, 'split_citations': 0,
              'additional_api_calls': 0, 'strict_validation_initially_passed': strict['valid']}
    strict['citation_recovery'] = report
    strict['citation_diagnostics'] = []
    if strict['valid']:
        return strict
    try:
        canonical = registry.validate_chunks(context_chunks, allowed_doc_ids=allowed_doc_ids)
    except (TypeError, ValueError, KeyError):
        return strict
    if not isinstance(payload, dict) or not isinstance(payload.get('claims'), list):
        return strict
    repaired = copy.deepcopy(payload)
    context = {chunk['chunk_id']: chunk for chunk in canonical}
    diagnostics = []
    corrections, splits = 0, 0
    for claim_index, claim in enumerate(repaired['claims'], 1):
        if not isinstance(claim, dict) or not isinstance(claim.get('citations'), list):
            continue
        replacement_citations = []
        for citation_index, citation in enumerate(claim['citations'], 1):
            if not isinstance(citation, dict):
                replacement_citations.append(citation)
                continue
            replacement_citations.append(citation)
            cid, quote = citation.get('chunk_id'), citation.get('quote')
            item = {'claim_index': claim_index, 'citation_index': citation_index,
                    'claim_doc_id': claim.get('doc_id'), 'chunk_id': cid,
                    'model_quote': quote, 'quote_is_model_data_not_a_verified_claim': True}
            marker = re.match(r'^\[([a-z_]+)\|(?:fact|notice_reference|conflict_candidate)\] ', str(claim.get('text', '')))
            item['claim_field_id'] = marker.group(1) if marker else None
            chunk = context.get(cid) if isinstance(cid, str) else None
            if chunk is None:
                matches = _literal_reference_candidates(context, claim, citation)
                item['outcome'] = 'chunk_not_in_supplied_context'
                if matches:
                    replacement_citations.pop()
                    replacement_citations.extend(
                        {'chunk_id': c, 'quote': quote}
                        for c in matches
                    )
                    item.update(
                        outcome='reference_rebuilt_from_literal_matches',
                        original_chunk_id=cid,
                        candidate_chunk_ids=matches,
                        reference_rebuilt=True,
                        requires_reference_review=True,
                        intended_original_location_inferred=False,
                        reference_quote_unchanged=True
                    )
                    corrections += 1
            elif claim.get('doc_id') != chunk['doc_id']:
                item['outcome'] = 'claim_and_chunk_document_differ'
            elif not isinstance(quote, str) or not quote.strip():
                item['outcome'] = 'invalid_quote'
            elif quote in chunk['text']:
                item['outcome'] = 'already_exact'
            else:
                recovered, outcome = _recover_whitespace(chunk['text'], quote)
                item['outcome'] = outcome
                if recovered is not None:
                    # Replacement is a literal source substring, not a normalized citation.
                    citation['quote'] = recovered['quote']
                    item['restored_source_quote'] = recovered['quote']
                    item['source_chunk_start'] = recovered['quote_start']
                    item['source_chunk_end'] = recovered['quote_end']
                    corrections += 1
                else:
                    item['cited_chunk_preview'] = chunk['text'][:300]
                    if outcome == 'non_whitespace_characters_differ' and set(citation) == {'chunk_id','quote'}:
                        fragments, split_outcome = _recover_line_fragments(chunk['text'], quote)
                        item['split_check'] = split_outcome
                        if fragments is not None:
                            replacement_citations.pop()
                            replacement_citations.extend({'chunk_id':cid,'quote':f['quote']} for f in fragments)
                            item['outcome'] = split_outcome
                            item['fragments'] = fragments
                            gaps = []
                            for first, second in zip(fragments, fragments[1:]):
                                left, right = first['quote_end'], second['quote_start']
                                gaps.append({'start': left, 'end': right,
                                             'text': chunk['text'][left:right],
                                             'preview': chunk['text'][left:right][:300]})
                            item['omitted_source_gap'] = gaps[0]
                            item['omitted_source_gaps'] = gaps
                            item['fragment_count'] = len(fragments)
                            item['requires_conditions_review'] = True
                            item['display_as_separate_quotes'] = True
                            corrections += 1
                            splits += 1
            diagnostics.append(item)
        claim['citations'] = replacement_citations
    validated = registry.validate_answer(repaired, canonical, allowed_doc_ids=allowed_doc_ids)
    validated['citation_recovery'] = {**report, 'corrected_citations': corrections, 'split_citations':splits, 'reference_rebindings': sum(bool(d.get('reference_rebuilt')) for d in diagnostics)}
    validated['citation_diagnostics'] = diagnostics
    return validated


def _cached_response(root, saved):
    """Read only the matching cache row, with checksum and original input checks."""
    fingerprint = (saved.get('generation') or {}).get('request_fingerprint')
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError('저장 결과에 요청 지문이 없습니다.')
    cache_path = Path(root) / 'phase4_outputs/generation_cache/grounded_responses.sqlite3'
    if not cache_path.is_file():
        raise FileNotFoundError('이 프로젝트의 기존 모델 응답 캐시가 없습니다. 추가 API 호출은 하지 않았습니다.')
    with sqlite3.connect(cache_path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        row = connection.execute('SELECT request_json,response_json,checksum FROM responses WHERE fingerprint=?',
                                 (fingerprint,)).fetchone()
    if row is None:
        raise ValueError('이 결과와 일치하는 캐시 응답이 없습니다. 추가 API 호출은 하지 않았습니다.')
    if hashlib.sha256(row[0].encode()).hexdigest() != fingerprint:
        raise ValueError('캐시 요청 지문이 다릅니다.')
    if hashlib.sha256((row[0] + '\n' + row[1]).encode()).hexdigest() != row[2]:
        raise ValueError('캐시 응답 무결성 검사에 실패했습니다.')
    request, response = json.loads(row[0]), json.loads(row[1])
    if response.get('request_fingerprint') != fingerprint:
        raise ValueError('캐시 응답이 다른 요청을 가리킵니다.')
    mode = response.get('execution_mode')
    if mode not in {'live_openai','injected_test_client'} or request.get('execution_namespace') != mode:
        raise ValueError('캐시 요청과 응답의 실제/모의 실행 구분이 다릅니다.')
    user = json.loads(request['request']['input'][1]['content'])
    expected_task = 'comparison' if saved.get('retrieval', {}).get('comparison') else 'qa'
    if user.get('task') != expected_task or user.get('question') != saved.get('retrieval', {}).get('standalone_query'):
        raise ValueError('캐시의 작업 또는 질문이 저장 결과와 다릅니다.')
    evidence = user.get('evidence', [])
    chunks = saved.get('source_chunks', [])
    if len(evidence) != len(chunks):
        raise ValueError('캐시와 저장 결과의 입력 근거 개수가 다릅니다.')
    for original, chunk in zip(evidence, chunks):
        for key in ['chunk_id','doc_id','sha256','text']:
            if original.get(key) != chunk.get(key):
                raise ValueError('캐시와 저장 결과의 입력 근거가 다릅니다.')
    generation = saved.get('generation') or {}
    if response.get('execution_mode') != generation.get('execution_mode') or response.get('live_api') != generation.get('live_api'):
        raise ValueError('캐시 응답의 모의/실제 실행 구분이 다릅니다.')
    return response


def replay_saved_result(root, result_path, *, registry=None, persist=True):
    """Revalidate an existing response without API key, SDK or network calls."""
    root, path = Path(root).resolve(), Path(result_path).resolve()
    saved = json.loads(path.read_text(encoding='utf-8'))
    response = _cached_response(root, saved)
    bootstrap_index = None
    if registry is None:
        from .runtime import bootstrap
        app = bootstrap(root)
        registry, bootstrap_index = app['registry'], app['index']
    try:
        checked = validate_with_quote_recovery(registry, response['payload'], saved['source_chunks'],
                                               allowed_doc_ids=saved.get('active_doc_ids') or None)
        result = copy.deepcopy(saved)
        result['validation'] = checked
        result['claims'] = checked['claims']
        result['missing_information'] = checked['missing_information']
        result['status'] = checked['status'] if checked['valid'] else 'not_generated'
        result['execution_status'] = 'completed' if checked['valid'] else 'validation_failed'
        result['answer_generated'] = bool(checked['valid'] and checked['claims'])
        result['markdown'] = registry.render(checked)
        if checked['valid'] and saved.get('retrieval', {}).get('comparison'):
            from .rag import _comparison_markdown
            scopes = saved.get('active_doc_ids') or []
            cited = {claim['doc_id'] for claim in checked['claims']}
            missing = [doc_id for doc_id in scopes if doc_id not in cited]
            if missing:
                checked['status'] = 'partial' if checked['claims'] else 'insufficient_evidence'
                checked['missing_information'].extend(f'선택 문서 {did}: 검증된 비교 근거 미확보' for did in missing)
                result['status'], result['missing_information'] = checked['status'],checked['missing_information']
            catalog = [{'doc_id': did, 'file_name': registry.documents_by_id[did].get('file_name',did)}
                       for did in scopes]
            result['markdown'] = _comparison_markdown(checked, scopes, catalog)
        result['replay'] = {'additional_api_calls': 0, 'original_result_path': str(path),
                            'scope': 'Offline source revalidation of the original cached model response; no new semantic quality measurement.'}
        result['generation']['original_usage'] = response.get('original_usage', response.get('usage', {}))
        result['generation']['usage'] = {'api_calls':0,'input_tokens':0,'output_tokens':0,
                                          'total_tokens':0,'cached_input_tokens':0,'usage_reported':True}
        result['generation']['cache_hit'] = True
        result['generation']['current_execution_mode'] = 'offline_source_revalidation'
        result['generation']['request_latency_ms'] = 0.0
        result['latency_ms'] = None
        if persist:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            destination = path.with_name(path.stem + '_revalidated_' + stamp + '.json')
            with destination.open('x',encoding='utf-8') as handle:
                json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
                handle.write('\n')
            result['revalidated_result_path'] = str(destination)
        return result
    finally:
        if bootstrap_index is not None:
            bootstrap_index.close()
