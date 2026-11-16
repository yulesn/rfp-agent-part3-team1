"""Optional read-only import of exact, fully valid legacy map responses.

Legacy request identity is preserved. A candidate is not a response to the new
SOURCE-unit transport and never receives a new request fingerprint. No recovery,
guessing, SDK import, API request, or cache mutation occurs here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3


LEGACY_TASK = 'phase5_field_summary_map_v1'
_EVIDENCE_KEYS = ('chunk_id', 'doc_id', 'sha256', 'file_name', 'locator',
                  'kind', 'source_method', 'quality_flags', 'text')
_MAX_ROWS = 10000
_MAX_ROW_CHARS = 12000000


def _digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _digest(value):
    return _digest_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False))


class _Skip(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Skip(reason)


def scan_legacy_batches(env, model):
    """Return source-validated legacy candidates and safe rejection reasons.

    ``source_start`` is inclusive and ``source_end`` exclusive in the current
    canonical document's ordered chunk list. Candidates can overlap; callers
    must choose a non-overlapping schedule and freeze its original fingerprints.
    Absence/corruption is a report state rather than an implicit API fallback.
    """
    from rfp_rag.generation.llm import (LLMConfig, LLMError, PROMPT_VERSION,
                               SYSTEM_INSTRUCTIONS, _parse_json, _validate_payload)
    from rfp_rag.summary_fields.fields import FIELD_QUESTION, FIELD_SCHEMA, parse_field_claim

    root = Path(env['root']).expanduser().resolve()
    path = root/'phase4_outputs/generation_cache/grounded_responses.sqlite3'
    report = {'version': 'phase5-verified-legacy-import-v1', 'status': 'cache_absent',
              'candidates': [], 'skipped': [], 'rows_inspected': 0,
              'candidate_count': 0, 'scan_fingerprint': _digest([]),
              'additional_api_calls': 0, 'cache_modified': False,
              'semantic_quality_verified': False}
    if not isinstance(model, str) or not model.strip():
        raise ValueError('기존 응답을 대조할 생성 모델 이름이 필요합니다.')
    if not path.is_file():
        return report
    registry = env['registry']
    documents = {}

    def current_document(did):
        if did not in documents:
            chunks = list(registry.validate_chunks(
                env['index'].get_document_chunks(did), allowed_doc_ids=[did]))
            documents[did] = (chunks, {c['chunk_id']: i for i, c in enumerate(chunks)})
        return documents[did]

    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as connection:
            cursor = connection.execute(
                'SELECT fingerprint,request_json,response_json,checksum '
                'FROM responses ORDER BY fingerprint LIMIT ?', (_MAX_ROWS+1,))
            for row_number, row in enumerate(cursor):
                if row_number == _MAX_ROWS:
                    report['skipped'].append({'reason': 'legacy_row_limit_reached'})
                    break
                report['rows_inspected'] += 1
                fp, request_text, response_text, checksum = row
                safe_fp = fp if isinstance(fp, str) and re.fullmatch(r'[0-9a-f]{64}', fp) else None
                try:
                    _require(safe_fp is not None, 'invalid_fingerprint')
                    _require(all(isinstance(v, str) and len(v) <= _MAX_ROW_CHARS
                                 for v in (request_text, response_text, checksum)), 'invalid_or_oversized_row')
                    _require(_digest_text(request_text) == fp, 'request_fingerprint_mismatch')
                    _require(_digest_text(request_text+'\n'+response_text) == checksum, 'cache_checksum_mismatch')
                    exact, response = _parse_json(request_text), _parse_json(response_text)
                    _require(isinstance(exact, dict) and isinstance(response, dict), 'invalid_cache_shape')
                    _require(exact.get('prompt_version') == PROMPT_VERSION, 'legacy_prompt_version_mismatch')
                    _require(exact.get('execution_namespace') == 'live_openai'
                             and response.get('execution_mode') == 'live_openai'
                             and response.get('live_api') is True, 'not_actual_model_origin')
                    _require(response.get('request_fingerprint') == fp, 'response_fingerprint_mismatch')
                    config = exact.get('config')
                    _require(isinstance(config, dict), 'invalid_legacy_config')
                    LLMConfig(**config)  # Pure validation only, never a model client.
                    _require(config.get('enabled') is True, 'legacy_generation_not_enabled')
                    body = exact.get('request')
                    _require(isinstance(body, dict) and body.get('model') == model
                             and config.get('model') == model, 'model_mismatch')
                    _require(body.get('max_output_tokens') == config.get('max_output_tokens')
                             and body.get('reasoning') == {'effort': config.get('reasoning_effort')},
                             'request_config_mismatch')
                    _require(body.get('store') is False, 'unexpected_legacy_request')
                    inputs = body.get('input')
                    _require(isinstance(inputs, list) and len(inputs) == 2, 'invalid_legacy_input')
                    _require(inputs[0] == {'role': 'system', 'content': SYSTEM_INSTRUCTIONS}
                             and inputs[1].get('role') == 'user', 'legacy_instructions_mismatch')
                    user = _parse_json(inputs[1].get('content'))
                    _require(user.get('task') == LEGACY_TASK, 'not_legacy_summary_task')
                    _require(user.get('question') == FIELD_QUESTION, 'legacy_question_mismatch')
                    fmt = body.get('text', {}).get('format', {})
                    _require(fmt.get('type') == 'json_schema' and fmt.get('strict') is True
                             and fmt.get('schema') == FIELD_SCHEMA, 'legacy_schema_mismatch')
                    evidence = user.get('evidence')
                    _require(isinstance(evidence, list) and evidence, 'empty_legacy_evidence')
                    _require(all(isinstance(c, dict) and set(c) == set(_EVIDENCE_KEYS)
                                 for c in evidence), 'invalid_legacy_evidence')
                    doc_ids = {c['doc_id'] for c in evidence}
                    _require(len(doc_ids) == 1, 'mixed_document_legacy_batch')
                    did = next(iter(doc_ids))
                    _require(isinstance(did, str) and did in registry.source_hashes, 'unknown_canonical_document')
                    chunks, position = current_document(did)
                    ids = [c['chunk_id'] for c in evidence]
                    _require(all(isinstance(cid, str) and cid in position for cid in ids)
                             and len(ids) == len(set(ids)), 'unknown_or_duplicate_legacy_chunks')
                    start = position[ids[0]]
                    end = start+len(ids)
                    current = chunks[start:end]
                    _require([c['chunk_id'] for c in current] == ids, 'legacy_chunk_order_or_interval_mismatch')
                    for old, source in zip(evidence, current):
                        _require(old == {k: source.get(k) for k in _EVIDENCE_KEYS},
                                 'legacy_source_content_or_version_mismatch')
                    payload = _validate_payload(response.get('payload'))
                    for claim in payload['claims']:
                        parse_field_claim(claim)
                    validated = registry.validate_answer(payload, current, allowed_doc_ids=[did])
                    _require(validated.get('valid') is True, 'legacy_citation_validation_failed')
                    candidate = {'doc_id': did, 'source_sha256': registry.source_hashes[did],
                        'source_chunk_ids': ids, 'source_start': start, 'source_end': end,
                        'legacy_request_fingerprint': fp,
                        'payload': copy.deepcopy(payload),
                        'payload_fingerprint': _digest(payload),
                        'source_context_fingerprint': _digest(evidence),
                        'validated_claims': validated['claims'],
                        'missing_information': list(validated.get('missing_information', [])),
                        'original_usage': copy.deepcopy(response.get('original_usage', response.get('usage', {}))),
                        'origin': 'legacy_literal_cache', 'execution_mode': 'live_openai',
                        'live_api': True, 'model': model, 'additional_api_calls': 0,
                        'semantic_quality_verified': False,
                        'scope': 'Original legacy request only; never relabelled as SOURCE-unit output.'}
                    candidate['candidate_fingerprint'] = _digest({k: candidate[k] for k in (
                        'doc_id', 'source_sha256', 'source_chunk_ids', 'legacy_request_fingerprint',
                        'payload_fingerprint', 'source_context_fingerprint', 'origin', 'model')})
                    report['candidates'].append(candidate)
                except _Skip as error:
                    report['skipped'].append({'legacy_request_fingerprint': safe_fp, 'reason': str(error)})
                except (ValueError, TypeError, KeyError, AttributeError, LLMError, RecursionError):
                    report['skipped'].append({'legacy_request_fingerprint': safe_fp,
                                              'reason': 'invalid_legacy_record'})
        report['status'] = 'scanned'
    except (sqlite3.Error, OSError):
        report['status'] = 'cache_unreadable'
        report['skipped'].append({'reason': 'legacy_cache_read_failed'})
    report['candidate_count'] = len(report['candidates'])
    report['scan_fingerprint'] = _digest([c['candidate_fingerprint'] for c in report['candidates']])
    return report
