"""Answer-bound source review. No model calls or automatic semantic approval."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import uuid


VERSION = 'phase5-source-review-final-v1'
FIELDS = (
    ('project_name', '사업명'), ('issuing_agency', '발주기관'), ('purpose', '목적'),
    ('scope_requirements', '주요 범위·필수요건'), ('budget_vat', '예산·VAT'),
    ('duration', '수행기간'), ('eligibility', '참가·수행자격'),
    ('submission_documents', '입찰 제출서류'), ('submission_deadline', '제출기한'),
    ('submission_method', '제출방식'), ('evaluation', '평가기준'),
)
JUDGEMENTS = ('정확성', '누락없음', '조건보존')
COMPARISON_BASE = ('항목ID', '항목', '기준표구분', '기준내용초안', '필수조건',
                   '실제요약', '확인상태', '기준원문위치', '검증된원문인용', '거절주장수')
COMPARISON_COLUMNS = (*COMPARISON_BASE, *JUDGEMENTS, '메모')
REJECTION_BASE = ('거절ID', '배치ID', '항목', '거절사유', '검증실패모델주장_진단용')
REJECTION_COLUMNS = (*REJECTION_BASE, '판정', '메모')
FAILED_BATCH_STATES = {
    'planned', 'awaiting_api', 'budget_blocked', 'interrupted', 'api_failed',
    'cache_invalid', 'validation_failed', 'error', 'failed', 'incomplete',
    'skipped', 'not_started',
}
UNPROCESSED_STATES = {
    'not_processed', 'unprocessed', 'not_started', 'awaiting_api',
    'execution_failed', 'incomplete',
}


def _digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def answer_fingerprint(result):
    """Bind accepted/rejected content, source versions and execution origin.

    Tokens, cost, latency, file locations and wall-clock timestamps do not
    alter a content review. Source quote ranges and canonical IDs do.
    """
    if not isinstance(result, dict):
        raise TypeError('실행 결과는 dict여야 합니다.')
    keys = ('doc_id', 'source_sha256', 'run_fingerprint', 'status',
            'execution_status', 'claims', 'fields', 'source_chunk_ids',
            'input_received_chunk_ids', 'checked_chunk_ids',
            'input_chunk_coverage', 'checked_chunk_coverage',
            'rejected_claims', 'model_origin')
    value = {key: result.get(key) for key in keys}
    batch_keys = ('batch_id', 'chunk_ids', 'source_chunk_ids', 'status',
                  'execution_status', 'execution_mode', 'live_api', 'model_origin',
                  'request_fingerprint', 'transport_fingerprint',
                  'binding_fingerprint', 'input_received', 'checked',
                  'accepted_claim_count', 'rejected_claim_count', 'rejection_ids',
                  'api_failed', 'execution_failed')
    value['batch_outcomes'] = [{key: b.get(key) for key in batch_keys}
                              if isinstance(b, dict) else b
                              for b in result.get('batch_outcomes', [])]
    return _digest(value)


def _safe_text(value):
    text = '' if value is None else str(value)
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def _safe_claim(claim, did, sha, *, field_projection=False):
    if (not isinstance(did, str) or not did or not isinstance(sha, str)
            or len(sha) != 64 or any(c not in '0123456789abcdefABCDEF' for c in sha)):
        return False
    claim_doc = claim.get('doc_id', did) if isinstance(claim, dict) and field_projection else (
        claim.get('doc_id') if isinstance(claim, dict) else None)
    if (not isinstance(claim, dict) or claim_doc != did
            or not isinstance(claim.get('text'), str) or not claim['text'].strip()):
        return False
    citations = claim.get('citations')
    return isinstance(citations, list) and bool(citations) and all(
        isinstance(c, dict) and c.get('provenance_verified') is True
        and c.get('doc_id') == did and c.get('sha256') == sha
        and isinstance(c.get('chunk_id'), str) and bool(c['chunk_id'])
        and isinstance(c.get('quote'), str) and bool(c['quote'].strip())
        for c in citations)


def _field_map(result):
    fields = result.get('fields') or []
    if not isinstance(fields, list):
        raise ValueError('요약 항목 목록의 형식이 잘못됐습니다.')
    known = dict(FIELDS)
    mapping = {}
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError('요약 항목의 형식이 잘못됐습니다.')
        key = field.get('field_id')
        if key not in known or key in mapping:
            raise ValueError('요약 항목 ID가 중복되거나 지원하지 않는 값입니다.')
        mapping[key] = field
    return mapping


def _reference_snapshot(result, references):
    if references is None:
        return []
    if not isinstance(references, dict):
        raise TypeError('기준표 자료는 dict 또는 None이어야 합니다.')
    did, sha = result.get('doc_id'), result.get('source_sha256')
    samples = [s for s in references.get('samples', []) if s.get('doc_id') == did]
    if len(samples) > 1:
        raise ValueError('기준표 문서 ID가 중복됐습니다.')
    if samples and samples[0].get('source_sha256') not in (None, sha):
        raise ValueError('기준표와 현재 원문의 SHA가 다릅니다.')
    sid = samples[0].get('sample_id') if samples else None
    selected = []
    for row in references.get('rows', []):
        if (sid is not None and row.get('sample_id') == sid) or (row.get('doc_id') is not None and row.get('doc_id') == did):
            selected.append({k: copy.deepcopy(row.get(k, '')) for k in
                             ('field_id', 'label', 'expected', 'conditions', 'source_reference')})
    ids = [r['field_id'] for r in selected]
    if len(ids) != len(set(ids)) or set(ids) - dict(FIELDS).keys():
        raise ValueError('기준표 항목 ID가 중복되거나 지원하지 않는 값입니다.')
    return selected


def _rejections(result):
    values = result.get('rejected_claims') or []
    if not isinstance(values, list):
        raise ValueError('거절 주장 목록의 형식이 잘못됐습니다.')
    ids = []
    for r in values:
        if not isinstance(r, dict) or not isinstance(r.get('rejection_id'), str) or not r['rejection_id'].strip():
            raise ValueError('거절 주장에는 고유한 rejection_id가 필요합니다.')
        ids.append(r['rejection_id'])
    if len(ids) != len(set(ids)):
        raise ValueError('거절 주장 ID가 중복됐습니다.')
    return values


def _templates(result, reference_snapshot):
    did, sha = result.get('doc_id'), result.get('source_sha256')
    fields, references = _field_map(result), {r['field_id']: r for r in reference_snapshot}
    rejected = _rejections(result)
    comparisons = []
    for field_id, label in FIELDS:
        field, ref = fields.get(field_id, {}), references.get(field_id, {})
        claims = [c for c in field.get('claims', []) if _safe_claim(c, did, sha, field_projection=True)]
        quotes = []
        for claim in claims:
            for cite in claim['citations']:
                text = ' | '.join(str(cite.get(k, '')) for k in ('file_name', 'locator', 'quote'))
                if text not in quotes:
                    quotes.append(text)
        count = sum(r.get('field_id') == field_id for r in rejected)
        actual = '\n'.join(c['text'] for c in claims)
        if not actual:
            actual = '검증된 주장을 표시할 수 없습니다.'
        notes = list(dict.fromkeys(n for n in field.get('notes', [])
                                   if isinstance(n, str) and n.strip()))
        if notes:
            actual += '\n추가 확인(모델 표시):\n' + '\n'.join(notes)
        values = (field_id, label,
                  '기준표 초안·원문 대조 필요' if ref else '원문 직접 대조',
                  ref.get('expected', ''), ref.get('conditions', ''), actual,
                  field.get('state', 'not_processed'), ref.get('source_reference', ''),
                  '\n'.join(quotes), count)
        row = {key: _safe_text(value) for key, value in zip(COMPARISON_BASE, values)}
        row.update({key: '' for key in (*JUDGEMENTS, '메모')})
        comparisons.append(row)
    rejections = []
    for item in rejected:
        raw = item.get('raw_claim')
        text = raw.get('text', '') if isinstance(raw, dict) else raw if isinstance(raw, str) else ''
        values = (item['rejection_id'], item.get('batch_id', ''),
                  dict(FIELDS).get(item.get('field_id'), item.get('field_id') or '항목 미확정'),
                  item.get('reason', ''), text or '주장문을 확인할 수 없습니다.')
        row = {key: _safe_text(value) for key, value in zip(REJECTION_BASE, values)}
        row.update({'판정': '', '메모': ''})
        rejections.append(row)
    return comparisons, rejections


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _write_csv_if_missing(path, columns, rows):
    path = Path(path)
    if path.exists():
        return
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # Atomic create; never replace a concurrent reviewer save.
        except FileExistsError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def create_review_files(folder, result, references=None):
    """Create immutable answer columns and blank reviewer cells; never refill CSV."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = _reference_snapshot(result, references)
    comparison, rejections = _templates(result, snapshot)
    paths = {'comparison_csv': folder/'comparison.csv',
             'rejection_review_csv': folder/'rejection_review.csv',
             'binding_json': folder/'review_binding.json'}
    binding = {'version': VERSION, 'answer_fingerprint': answer_fingerprint(result),
        'doc_id': result.get('doc_id'), 'source_sha256': result.get('source_sha256'),
        'run_fingerprint': result.get('run_fingerprint'),
        'reference_snapshot': snapshot, 'reference_fingerprint': _digest(snapshot),
        'comparison_base_fingerprint': _digest([{k: r[k] for k in COMPARISON_BASE} for r in comparison]),
        'rejection_base_fingerprint': _digest([{k: r[k] for k in REJECTION_BASE} for r in rejections]),
        'semantic_quality_automatically_verified': False}
    path = paths['binding_json']
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != binding:
            raise ValueError('검토표가 다른 답변·원문·기준표에 연결돼 있습니다. 기존 입력을 보존하고 중단합니다.')
    else:
        if paths['comparison_csv'].exists() or paths['rejection_review_csv'].exists():
            raise ValueError('기존 CSV의 답변 연결 기록이 없습니다. 검토 입력을 보존하고 중단합니다.')
        _atomic_json(path, binding)
    _write_csv_if_missing(paths['comparison_csv'], COMPARISON_COLUMNS, comparison)
    _write_csv_if_missing(paths['rejection_review_csv'], REJECTION_COLUMNS, rejections)
    return {key: str(value) for key, value in paths.items()}


def _read_and_check(path, columns, base_columns, expected, identifier):
    if not path.is_file():
        return [], ['review_file_missing:' + path.name]
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = reader.fieldnames
    errors = []
    if headers != list(columns):
        errors.append('review_columns_changed:' + path.name)
    if any(None in row for row in rows):
        errors.append('review_row_has_extra_columns:' + path.name)
    keys = [r.get(identifier) for r in rows]
    expected_by_id = {r[identifier]: r for r in expected}
    if len(keys) != len(set(keys)) or set(keys) != set(expected_by_id):
        errors.append('review_rows_changed:' + path.name)
        return rows, errors
    for row in rows:
        original = expected_by_id[row[identifier]]
        if any(row.get(k) != original[k] for k in base_columns):
            errors.append('readonly_cells_changed:' + path.name + ':' + row[identifier])
    return rows, errors


def _technical_blockers(result):
    blockers = []
    did, sha = result.get('doc_id'), result.get('source_sha256')
    if not isinstance(did, str) or not did or not isinstance(sha, str) or not sha:
        blockers.append('source_identity_missing')
    if result.get('execution_status') not in {'completed', 'completed_with_rejections'}:
        blockers.append('execution_not_completed')
    if result.get('status') not in {'answered', 'partial'}:
        blockers.append('answer_status_not_usable')
    expected = result.get('source_chunk_ids') or []
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(cid, str) or not cid for cid in expected)
            or len(expected) != len(set(expected))):
        blockers.append('source_chunk_list_invalid')
    for key in ('input_received_chunk_ids', 'checked_chunk_ids'):
        if result.get(key) != expected:
            blockers.append(key + '_incomplete')
    for key in ('input_chunk_coverage', 'checked_chunk_coverage'):
        value = result.get(key)
        if type(value) not in (int, float) or value != 1.0:
            blockers.append(key + '_not_complete')
    claims = result.get('claims') or []
    if not claims or not all(_safe_claim(c, did, sha) for c in claims):
        blockers.append('validated_answer_claims_missing')
    if any(q.get('chunk_id') not in expected for c in claims if isinstance(c, dict)
           for q in c.get('citations', []) if isinstance(q, dict)):
        blockers.append('citation_outside_document_chunks')
    if (result.get('model_origin') or {}).get('all_live_or_live_cache') is not True:
        blockers.append('live_model_origin_not_confirmed')
    fields = _field_map(result)
    if len(fields) != 11 or any(not isinstance(f.get('state'), str) or not f['state']
                              or f['state'] in UNPROCESSED_STATES for f in fields.values()):
        blockers.append('unprocessed_fields')
    if any(f.get('label') != dict(FIELDS)[fid] for fid, f in fields.items()):
        blockers.append('field_labels_mismatch')
    if any(not _safe_claim(c, did, sha, field_projection=True) for f in fields.values() for c in f.get('claims', [])):
        blockers.append('unverified_field_claims')
    batches = result.get('batch_outcomes') or []
    if not batches:
        blockers.append('batch_origin_missing')
    elif any(not isinstance(b, dict) or b.get('status') not in {'completed', 'completed_with_rejections'}
             or b.get('execution_status') in FAILED_BATCH_STATES
             or b.get('api_failed') is True or b.get('execution_failed') is True for b in batches):
        blockers.append('batch_execution_failure')
    return list(dict.fromkeys(blockers))


def assess_review_files(folder, result, reviewer=''):
    """Assess one document's source review, with a separate rejection disposition."""
    folder = Path(folder)
    errors, pending, failed, positive = [], [], [], 0
    try:
        binding = json.loads((folder/'review_binding.json').read_text(encoding='utf-8'))
        snapshot = binding['reference_snapshot']
        comparison, rejections = _templates(result, snapshot)
        if binding.get('version') != VERSION:
            errors.append('review_version_mismatch')
        if binding.get('answer_fingerprint') != answer_fingerprint(result):
            errors.append('stale_answer_or_source_review')
        if binding.get('reference_fingerprint') != _digest(snapshot):
            errors.append('reference_snapshot_changed')
        if binding.get('comparison_base_fingerprint') != _digest([{k: r[k] for k in COMPARISON_BASE} for r in comparison]):
            errors.append('comparison_binding_changed')
        if binding.get('rejection_base_fingerprint') != _digest([{k: r[k] for k in REJECTION_BASE} for r in rejections]):
            errors.append('rejection_binding_changed')
        rows, issues = _read_and_check(folder/'comparison.csv', COMPARISON_COLUMNS, COMPARISON_BASE, comparison, '항목ID')
        errors.extend(issues)
        rejected_rows, issues = _read_and_check(folder/'rejection_review.csv', REJECTION_COLUMNS, REJECTION_BASE, rejections, '거절ID')
        errors.extend(issues)
        for row in rows:
            for key in JUDGEMENTS:
                value = str(row.get(key, '')).strip().casefold()
                item = {'field_id': row.get('항목ID'), 'judgement': key}
                if value in {'통과', 'true'}:
                    positive += 1
                elif value in {'실패', 'false'}:
                    failed.append(item)
                elif value in {'', '미검토'}:
                    pending.append(item)
                else:
                    errors.append('invalid_judgement:' + str(row.get('항목ID')) + ':' + key)
        for row in rejected_rows:
            value = str(row.get('판정', '')).strip()
            item = {'rejection_id': row.get('거절ID'), 'judgement': '거절 주장 처리'}
            if value == '제외 적절':
                positive += 1
            elif value == '요약 보완 필요':
                failed.append(item)
            elif value in {'', '미검토'}:
                pending.append(item)
            else:
                errors.append('invalid_rejection_disposition:' + str(row.get('거절ID')))
        blockers = _technical_blockers(result)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        errors.append('review_binding_or_result_invalid')
        blockers = ['review_not_bound']
    if not isinstance(reviewer, str) or not reviewer.strip():
        pending.append({'field_id': None, 'judgement': '검토자 이름'})
    status = ('invalid_review' if errors else 'review_failed' if failed else
              'review_pending' if pending else 'passed_human_review')
    accepted = status == 'passed_human_review' and not blockers
    return {'version': VERSION, 'status': status, 'case_accepted': accepted,
        'case_status': 'accepted' if accepted else 'technical_failed' if blockers else status,
        'reviewer': reviewer.strip() if isinstance(reviewer, str) else '',
        'technical_ready': not blockers, 'technical_blockers': blockers,
        'diagnostics': sorted(set(errors)), 'pending_judgements': pending,
        'failed_judgements': failed, 'positive_judgements': positive,
        'human_review_completed': status in {'passed_human_review', 'review_failed'},
        'semantic_quality_automatically_verified': False,
        'system_quality_verified': False,
        'scope': '이 답변과 원문에 연결된 사람 검토입니다. 전체 문서·시스템 성능 판정이 아닙니다.'}
