"""Shared Python entry points used by the CLI and every retained notebook."""
from __future__ import annotations

import os
from pathlib import Path

from .pilot.pilot import digest
from .pilot.validation import run_validation, validation_status
from .recovery.runner import RecoveryRun
from .summary_full.engine import FinalConfig
from .summary_content.pipeline import rebuild_document_summary
from .summary_content.storage import save_content_result


def require_api_key(env_file=None):
    if env_file is not None:
        from dotenv import load_dotenv
        load_dotenv(Path(env_file).expanduser(), override=False)
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        raise ValueError('실제 생성에는 OPENAI_API_KEY 환경변수가 필요합니다.')


def search(env, question, *, doc_ids, top_k=None):
    ids = env['registry'].resolve_doc_ids(list(doc_ids))
    if not ids:
        raise ValueError('검색 대상 문서를 선택하세요.')
    settings = {key: value for key, value in env['source_manifest']['settings']['retrieval'].items()
                if key in {'method', 'top_k', 'fetch_k', 'mmr_lambda'}}
    if top_k is not None:
        settings['top_k'] = top_k
    hits = env['index'].search(question, allowed_doc_ids=ids, **settings)
    env['registry'].validate_chunks(hits, allowed_doc_ids=ids)
    return hits


def submit_questions(pilot, questions, *, doc_ids, live=False, batch_id='notebook-01', env_file=None):
    """One retained pilot/client; stable IDs make identical cell replay idempotent."""
    if type(live) is not bool:
        raise TypeError('live는 True/False여야 합니다.')
    if live:
        require_api_key(env_file)
    pilot.configure_generation(enabled=live, approved_doc_ids=doc_ids if live else [])
    pilot.select(doc_ids, filters={})
    records = []
    for number, question in enumerate(questions):
        record = pilot.submit(question, request_id=f'{batch_id}:{live}:{number}')
        records.append(record)
        if pilot.halted:
            break
    return records


def summary_runner(env, *, run_id='summary-01', client=None):
    settings = env['source_manifest']['settings']
    generation = settings['generation']
    mapped = {'model': generation['model'], 'max_output_tokens': generation['max_output_tokens'],
              'reasoning_effort': generation['reasoning_effort'],
              'max_request_bytes': generation['max_request_bytes'],
              'batch_max_bytes': settings['summary']['max_batch_bytes'],
              'max_total_calls': generation['max_calls'],
              'max_total_input_bytes': generation['max_total_input_bytes']}
    return RecoveryRun(env, run_id=run_id, config=FinalConfig(**mapped), client=client)


def summarize(env, doc_ids, *, run_id='summary-01', live=False, env_file=None, client=None):
    if type(live) is not bool:
        raise TypeError('live는 True/False여야 합니다.')
    ids = env['registry'].resolve_doc_ids(list(doc_ids))
    if not ids:
        raise ValueError('요약 대상 문서를 선택하세요.')
    if live and client is None:
        require_api_key(env_file)
    runner = summary_runner(env, run_id=run_id, client=client)
    if live:
        runner.run(ids, allow_api=True)
    results = []
    for did in ids:
        raw = runner.result(did)
        document = env['registry'].documents_by_id[did]
        chunks = env['index'].get_document_chunks(did)
        derived = rebuild_document_summary(raw, document, chunks, env['registry'])
        save_content_result(runner.folder, raw, derived)
        results.append(derived)
    return runner, results


def validate(env, doc_ids, *, live=False, env_file=None, client=None):
    if live and client is None:
        require_api_key(env_file)
    return run_validation(env, list(doc_ids), enabled=live,
                          approved_doc_ids=list(doc_ids) if live else None, client=client)


def validate_once(env, doc_ids, *, runs, live=False, batch_id='validation-01', env_file=None, client=None):
    """Keep one validation run per explicit notebook batch and exact source scope."""
    if type(live) is not bool:
        raise TypeError('live는 True/False여야 합니다.')
    ids = env['registry'].resolve_doc_ids(list(doc_ids))
    signature = digest({'root': str(env['root']), 'ids': ids, 'live': live, 'batch_id': batch_id,
                        'sources': env['registry'].source_hashes,
                        'index': env['index'].get_metadata()['index_fingerprint']})
    if signature in runs:
        folder = runs[signature]
        return folder, validation_status(folder)
    folder, report = validate(env, ids, live=live, env_file=env_file, client=client)
    runs[signature] = folder
    return folder, report
