"""Versioned, zero-API derived summaries beside immutable generation records.

The frozen model outputs and their review forms are never rewritten here.
Content repair has its own identity, review cells, latest pointer and report.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import uuid

from rfp_rag.summary_full.catalog_audit import get_validated_document_chunks
from rfp_rag.summary_full.engine import read_json, render_result, write_json
from rfp_rag.summary_full.review import answer_fingerprint


VERSION = 'phase5-content-storage-v1'
BASE_COLUMNS = ('항목ID', '항목', '실제요약', '확인상태', '검증된원문인용')
JUDGEMENTS = ('정확성', '누락없음', '조건보존')
COLUMNS = (*BASE_COLUMNS, *JUDGEMENTS, '메모')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def code_fingerprint():
    """Bind the derived output to the complete running correction package."""
    return digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))
                   if not p.name.startswith('test_')})


def _safe_id(value):
    if (not isinstance(value, str) or not value or value in {'.', '..'}
            or '/' in value or '\\' in value or '\x00' in value):
        raise ValueError('문서 ID를 출력 경로로 사용할 수 없습니다.')
    return value


def _safe_text(value):
    text = str(value or '')
    return "'"+text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def _review_rows(result):
    rows = []
    for field in result.get('fields', []):
        quotes = []
        for claim in field.get('claims', []):
            for cite in claim.get('citations', []):
                text = str(cite.get('locator', ''))+' | '+str(cite.get('quote', ''))
                if text not in quotes:
                    quotes.append(text)
        row = dict(zip(BASE_COLUMNS, map(_safe_text, (
            field['field_id'], field['label'],
            '\n'.join(c.get('text', '') for c in field.get('claims', [])),
            field.get('state', 'not_processed'), '\n'.join(quotes)))))
        row.update({key:'' for key in (*JUDGEMENTS, '메모')})
        rows.append(row)
    return rows


def _write_review_if_missing(path, rows):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=COLUMNS)
            writer.writeheader();writer.writerows(rows)
            handle.flush();os.fsync(handle.fileno())
        try:
            os.link(temporary,path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def save_content_result(run_folder, raw_result, derived_result):
    """Save a correction revision without changing raw results or review input."""
    run_folder = Path(run_folder).resolve()
    did = _safe_id(raw_result.get('doc_id'))
    if derived_result.get('doc_id') != did or derived_result.get('source_sha256') != raw_result.get('source_sha256'):
        raise ValueError('원래 결과와 보정 결과의 문서·원문 버전이 다릅니다.')
    result = copy.deepcopy(derived_result)
    raw_fp = answer_fingerprint(raw_result)
    policy_fp = code_fingerprint()
    fingerprint = digest({'version':VERSION, 'input_result_fingerprint':raw_fp,
                          'content_code_fingerprint':policy_fp, 'result':result})
    result['content_storage'] = {'version':VERSION, 'content_fingerprint':fingerprint,
        'input_result_fingerprint':raw_fp, 'content_code_fingerprint':policy_fp,
        'api_calls':0, 'original_results_modified':False,
        'human_review_completed':False}
    folder = run_folder/'content_documents'/did/fingerprint[:24]
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'result.json'
    if result_path.exists() and read_json(result_path) != result:
        raise ValueError('보정 결과의 지문과 저장된 내용이 다릅니다.')
    if not result_path.exists():
        write_json(result_path, result)
    markdown = render_result(result)
    markdown += ('\n생성 결과와 원문을 문서 전체 맥락으로 다시 취합한 결과입니다. '
                 '원문에서 추가한 사실은 local_source_extraction으로 구분해 기록했습니다.\n')
    (folder/'answer.md').write_text(markdown, encoding='utf-8')
    rows = _review_rows(result)
    base_rows = [{k:r[k] for k in BASE_COLUMNS} for r in rows]
    binding = {'version':VERSION, 'content_fingerprint':fingerprint,
        'input_result_fingerprint':raw_fp, 'source_sha256':result['source_sha256'],
        'review_base_fingerprint':digest(base_rows),
        'human_review_completed':False, 'semantic_quality_automatically_verified':False}
    path = folder/'review_binding.json'
    if path.exists() and read_json(path) != binding:
        raise ValueError('기존 보정 검토표가 다른 결과에 연결되어 있습니다.')
    if not path.exists():
        write_json(path, binding)
    review_path = folder/'review.csv'
    if not review_path.exists():
        _write_review_if_missing(review_path,rows)
    write_json(run_folder/'content_documents'/did/'latest.json', {
        'folder':folder.relative_to(run_folder).as_posix(),
        'content_fingerprint':fingerprint, 'input_result_fingerprint':raw_fp})
    return result


def _safe_folder(root, relative, expected_parent):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('결과 경로는 실행 폴더 내부의 상대경로여야 합니다.')
    target = (root/relative).resolve()
    if not target.is_relative_to(expected_parent.resolve()) or target == expected_parent.resolve():
        raise ValueError('결과 경로가 해당 문서 폴더를 벗어납니다.')
    return target


def load_saved_result(run_folder, doc_id, *, content=False):
    """Load one strict latest pointer and reject traversal or swapped revisions."""
    root = Path(run_folder).resolve();did = _safe_id(doc_id)
    collection = 'content_documents' if content else 'documents'
    parent = root/collection/did
    pointer = parent/'latest.json'
    if not pointer.resolve().is_relative_to(root) or not parent.resolve().is_relative_to(root):
        raise ValueError('문서 경로가 실행 폴더를 벗어납니다.')
    latest = read_json(pointer)
    folder = _safe_folder(root, latest.get('folder'), parent)
    path = folder/'result.json'
    if not path.resolve().is_relative_to(folder):
        raise ValueError('결과 파일 경로가 문서 폴더를 벗어납니다.')
    result = read_json(path)
    if result.get('doc_id') != did:
        raise ValueError('문서 ID와 저장 결과가 다릅니다.')
    if content:
        metadata=result.get('content_storage') or {}
        fingerprint=metadata.get('content_fingerprint')
        payload=copy.deepcopy(result);payload.pop('content_storage',None)
        expected=digest({'version':VERSION,'input_result_fingerprint':metadata.get('input_result_fingerprint'),
            'content_code_fingerprint':metadata.get('content_code_fingerprint'),'result':payload})
        if not fingerprint or latest.get('content_fingerprint') != fingerprint or expected!=fingerprint:
            raise ValueError('보정 결과 지문이 latest 연결과 다릅니다.')
    elif answer_fingerprint(result) != latest.get('answer_fingerprint'):
        raise ValueError('생성 결과 지문이 latest 연결과 다릅니다.')
    frozen=root/'frozen.json'
    if not content and frozen.exists() and digest(read_json(frozen))!=result.get('run_fingerprint'):
        raise ValueError('저장 생성 결과가 다른 고정 실행에 연결되어 있습니다.')
    return result, folder


def assess_content_review(folder, result, reviewer=''):
    """Read only explicit human cells bound to this derived revision."""
    folder = Path(folder)
    expected = _review_rows(result)
    diagnostics = [];pending = [];failed = []
    try:
        binding = read_json(folder/'review_binding.json')
        content_fp = (result.get('content_storage') or {}).get('content_fingerprint')
        if not content_fp or binding.get('content_fingerprint') != content_fp:
            raise ValueError('review is bound to another content result')
        with (folder/'review.csv').open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle);rows = list(reader)
            if reader.fieldnames != list(COLUMNS):
                raise ValueError('review columns changed')
        expected_by_id = {r['항목ID']:r for r in expected}
        if len(rows) != len(expected) or {r.get('항목ID') for r in rows} != set(expected_by_id):
            raise ValueError('review rows changed')
        base_rows = [{k:r.get(k, '') for k in BASE_COLUMNS} for r in rows]
        if digest([{k:r[k] for k in BASE_COLUMNS} for r in expected]) != binding.get('review_base_fingerprint'):
            raise ValueError('review binding changed')
        for row in rows:
            if None in row or any(row.get(k) != expected_by_id[row['항목ID']][k] for k in BASE_COLUMNS):
                raise ValueError('read-only review values changed')
            for key in JUDGEMENTS:
                value = str(row.get(key, '')).strip().casefold()
                item = {'field_id':row['항목ID'], 'judgement':key}
                if value in {'실패','false'}:
                    failed.append(item)
                elif value in {'','미검토'}:
                    pending.append(item)
                elif value not in {'통과','true'}:
                    raise ValueError('unsupported review judgement')
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        diagnostics.append('invalid_or_stale_content_review')
    if not isinstance(reviewer, str) or not reviewer.strip():
        pending.append({'field_id':None, 'judgement':'검토자 이름'})
    technical_ready = (result.get('execution_status') in {'completed','completed_with_rejections'}
        and result.get('input_chunk_coverage') == 1.0 and result.get('checked_chunk_coverage') == 1.0
        and bool(result.get('claims')) and bool(result.get('answer_generated')))
    status = ('invalid_review' if diagnostics else 'review_failed' if failed else
              'review_pending' if pending else 'passed_human_review')
    return {'status':status, 'case_accepted':status=='passed_human_review' and technical_ready,
        'human_review_completed':status in {'passed_human_review','review_failed'},
        'technical_ready':technical_ready, 'pending_judgements':pending,
        'failed_judgements':failed, 'diagnostics':diagnostics,
        'reviewer':reviewer.strip() if isinstance(reviewer, str) else '',
        'semantic_quality_automatically_verified':False, 'system_quality_verified':False}


def saved_document_ids(run_folder):
    root = Path(run_folder).resolve()
    folder = root/'documents'
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and (p/'latest.json').is_file())


def repair_saved_results(env, run_folder, doc_ids=None):
    """Repair existing saved results only. No runner, SDK, key or API access."""
    from .pipeline import rebuild_document_summary
    root = Path(run_folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError('기존 실행 결과 폴더를 찾을 수 없습니다.')
    ids = saved_document_ids(root) if doc_ids is None else env['registry'].resolve_doc_ids(doc_ids)
    rows = []
    documents = {d['doc_id']:d for d in env['documents']}
    for did in ids:
        try:
            raw, raw_folder = load_saved_result(root, did)
            if did not in documents:
                raise ValueError('저장 결과 문서가 현재 원문 목록에 없습니다.')
            document = documents[did]
            if document.get('sha256') != raw.get('source_sha256'):
                raise ValueError('저장 결과의 원문 버전이 다릅니다.')
            chunks = get_validated_document_chunks(env, did)
            result = rebuild_document_summary(raw, document, chunks, env['registry'])
            result = save_content_result(root, raw, result)
            latest = read_json(root/'content_documents'/did/'latest.json')
            rows.append({'doc_id':did, 'status':'repaired',
                'generation_execution_status':raw.get('execution_status'),
                'content_fingerprint':result['content_storage']['content_fingerprint'],
                'raw_result_path':str(raw_folder/'result.json'),
                'result_path':str(root/latest['folder']/'result.json'),
                'answer_path':str(root/latest['folder']/'answer.md'),
                'review_path':str(root/latest['folder']/'review.csv'),
                'api_calls':0, 'human_review_completed':False})
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            rows.append({'doc_id':did, 'status':'error',
                'error':{'code':'saved_content_repair_failed','exception_type':type(error).__name__},
                'api_calls':0})
    report = {'version':VERSION, 'selected_saved_documents':len(ids),
        'repaired_documents':sum(r['status']=='repaired' for r in rows),
        'failed_documents':sum(r['status']=='error' for r in rows), 'documents':rows,
        'api_calls':0, 'generation_reexecuted':False, 'original_results_modified':False,
        'human_review_completed':False, 'all_documents_semantic_quality_verified':False,
        'output_folder':str(root/'content_documents')}
    write_json(root/'content_repair_report.json',report)
    return report
