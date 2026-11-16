"""Explicit, resumable web summaries of every chunk in one selected document.

Planning and reads are offline. A worker opens its own SQLite connection;
neither the HTTP thread's index nor its question-generation budget is shared.
The existing source-unit runner owns durable reservations, validation and cache
replay. One stable run per document/model prevents refreshes from spending again.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import threading

from rfp_rag.evaluation.evaluate import _load_summary
from rfp_rag.generation.llm import safe_llm_diagnostic
from rfp_rag.recovery.runner import RecoveryRun, load_api_key
from rfp_rag.summary_content.pipeline import rebuild_document_summary
from rfp_rag.summary_content.storage import save_content_result
from rfp_rag.summary_full.engine import FinalConfig, digest, write_json
from rfp_rag.workspace import open_workspace


VERSION = 'web-document-summary-v1'
COMPLETE = {'completed', 'completed_with_rejections'}


class SummaryInputError(ValueError):
    """Only fixed, user-facing messages, never provider bodies or source text."""


def summary_config(env):
    settings = env['source_manifest']['settings']
    generation = settings['generation']
    return FinalConfig(
        model=generation['model'], reasoning_effort=generation['reasoning_effort'],
        max_output_tokens=generation['max_output_tokens'],
        max_request_bytes=generation['max_request_bytes'],
        batch_max_bytes=settings['summary']['max_batch_bytes'],
        max_batches_per_document=settings['summary']['max_map_calls'],
        max_total_calls=settings['summary']['max_map_calls'],
        max_total_input_bytes=generation['max_total_input_bytes'])


def run_id(env, doc_id, config, *, injected=False):
    return 'web-summary-' + digest({
        'version': VERSION, 'doc_id': doc_id,
        'source_sha256': env['registry'].source_hashes[doc_id],
        'index_fingerprint': env['index'].get_metadata()['index_fingerprint'],
        'config': asdict(config), 'mode': 'test_double' if injected else 'live_openai'})[:40]


def progress(runner, doc_id):
    batches = runner.plan(doc_id)['batches']
    jobs = runner._journal()['jobs']
    selected = [jobs.get(batch['job_key'], {}) for batch in batches]
    usages = [attempt.get('usage') or {} for job in selected
              for attempt in job.get('attempts', [])]
    keys = ('api_calls', 'live_api_calls', 'mock_calls', 'failed_attempts',
            'reported_input_tokens', 'reported_output_tokens', 'reported_total_tokens',
            'usage_unknown_attempts')
    return {
        'total_batches': len(batches),
        'completed_batches': sum(job.get('status') in COMPLETE for job in selected),
        'failed_batches': sum(job.get('status') in {
            'api_failed', 'validation_failed', 'cache_invalid', 'interrupted'} for job in selected),
        'budget': runner.budget_usage(),
        'usage': {key: sum(usage.get(key, 0) for usage in usages) for key in keys}}


class ProgressRun(RecoveryRun):
    """Observe the existing atomic journal; do not change requests or validation."""

    def _save_journal(self, journal):
        super()._save_journal(journal)
        callback = getattr(self, 'on_progress', None)
        if callback is not None:
            callback()


class WebSummaries:
    def __init__(self, root, *, client=None):
        self.root = Path(root).resolve()
        self.client = client  # Test doubles only; real SDK stays inside RecoveryLLM.
        self._plans = {}
        self._jobs = {}
        self._current = None
        self._lock = threading.Lock()

    def current(self):
        with self._lock:
            return copy.deepcopy(self._jobs.get(self._current))

    def running(self):
        return (self.current() or {}).get('status') in {'queued', 'running'}

    def _runner(self, env, doc_id):
        config = summary_config(env)
        identifier = run_id(env, doc_id, config, injected=self.client is not None)
        return ProgressRun(env, run_id=identifier, config=config, client=self.client)

    def prepare(self, env, doc_id):
        did = env['registry'].resolve_doc_ids([doc_id])[0]
        if self.running():
            raise SummaryInputError('요약 생성이 진행 중입니다. 완료 후 다른 요청을 실행하세요.')
        runner = self._runner(env, did)
        # Revalidate saved batch bindings before treating them as already done.
        runner.result(did)
        check = runner.preflight([did])
        row = check['rows'][0]
        document = env['registry'].documents_by_id[did]
        can_start = (row['can_execute'] and check['within_remaining_budget']
                     and not row['failed_or_interrupted_batches']
                     and not runner._journal().get('service_blocker'))
        plan = {
            'version': VERSION, 'doc_id': did, 'file_name': document.get('file_name', did),
            'model': runner.config.model, 'run_id': runner.folder.name,
            'run_fingerprint': runner.fingerprint, 'plan_fingerprint': row['plan_fingerprint'],
            'chunk_count': row['chunk_count'], 'batch_count': row['batch_count'],
            'new_api_calls': check['uncached_calls'],
            'new_input_bytes': check['uncached_input_bytes'],
            'saved_batches': row['saved_batches'],
            'cached_batches': row['available_cached_batches'],
            'failed_batches': row['failed_or_interrupted_batches'],
            'can_start': bool(can_start), 'within_budget': check['within_remaining_budget'],
            'block_reasons': row['block_reasons'], 'budget': check['budget'],
            'api_calls_by_planning': 0,
            'budget_scope': 'This document/model summary; persistent across web sessions. Separate from questions.'}
        plan['plan_id'] = digest(plan)
        self._plans[plan['plan_id']] = plan
        return copy.deepcopy(plan)

    def saved(self, env, doc_id):
        did = env['registry'].resolve_doc_ids([doc_id])[0]
        config = summary_config(env)
        identifier = run_id(env, did, config, injected=self.client is not None)
        folder = self.root / 'phase5_outputs/final_source_units' / identifier
        result, reason = _load_summary(env, folder, did)
        if result is None:
            return {'status': 'unavailable', 'reason': reason, 'doc_id': did, 'api_calls': 0}
        # Never expose rejected model text or diagnostics as accepted summary content.
        visible = {key: copy.deepcopy(result[key]) for key in (
            'doc_id', 'source_sha256', 'status', 'execution_status', 'answer_generated',
            'fields', 'claims', 'missing_information', 'input_chunk_coverage',
            'checked_chunk_coverage', 'source_chunk_ids', 'content_storage') if key in result}
        visible.update(semantic_quality_verified=False, human_review_completed=False)
        return {'status': 'saved_summary', 'doc_id': did, 'result': visible,
                'api_calls': 0, 'summary_run': identifier,
                'semantic_quality_verified': False}

    def _publish(self, identifier, **values):
        with self._lock:
            job = self._jobs[identifier]
            job.update(copy.deepcopy(values))
            snapshot = copy.deepcopy(job)
        folder = self.root / 'phase5_outputs/final_source_units' / identifier
        write_json(folder / 'web_job.json', snapshot)

    def status(self, identifier):
        with self._lock:
            if not isinstance(identifier, str) or identifier not in self._jobs:
                raise SummaryInputError('요약 실행 ID를 확인하세요.')
            return copy.deepcopy(self._jobs[identifier])

    def start(self, env, doc_id, *, plan_id, allow_api=False):
        if type(allow_api) is not bool or not allow_api:
            raise SummaryInputError('요약용 실제 API 실행을 명시적으로 허용하세요.')
        did = env['registry'].resolve_doc_ids([doc_id])[0]
        plan = self._plans.get(plan_id) if isinstance(plan_id, str) else None
        if plan is None or plan['doc_id'] != did:
            raise SummaryInputError('문서를 다시 선택 적용하고 요약 계획을 확인하세요.')
        current = self.current()
        if current and current['job_id'] == plan['run_id']:
            return current  # Replayed POST, double click, or an already finished run.
        if self.running():
            raise SummaryInputError('다른 요약이 진행 중입니다. 완료 후 실행하세요.')
        if not plan['can_start']:
            raise SummaryInputError('요약 계획의 호출 한도·실패 기록을 확인하세요. 자동 재시도하지 않습니다.')
        if plan['new_api_calls'] and self.client is None:
            load_api_key(self.root)
        job = {'version': VERSION, 'job_id': plan['run_id'], 'doc_id': did,
               'model': plan['model'], 'status': 'queued', 'plan_id': plan_id,
               'total_batches': plan['batch_count'], 'completed_batches': plan['saved_batches'],
               'failed_batches': 0, 'budget': plan['budget'], 'usage': {},
               'new_api_calls': 0, 'automatic_retry': False,
               'semantic_quality_verified': False}
        with self._lock:
            self._jobs[job['job_id']] = job
            self._current = job['job_id']
        # Persist permission before the worker can enter the paid runner.
        self._publish(job['job_id'], approved_plan=plan)
        thread = threading.Thread(target=self._execute, args=(plan,), daemon=True,
                                  name='rfp-document-summary')
        thread.start()
        return self.status(job['job_id'])

    def _execute(self, plan):
        identifier, did = plan['run_id'], plan['doc_id']
        try:
            # Own connection, same verified index identity as planning. C/D also
            # verify their saved vectors, but never search or embed a query.
            with open_workspace(self.root) as env:
                runner = self._runner(env, did)
                if runner.fingerprint != plan['run_fingerprint']:
                    raise SummaryInputError('계획 이후 원문·모델·구현이 바뀌었습니다. 다시 확인하세요.')
                check = runner.preflight([did])
                if (check['rows'][0]['plan_fingerprint'] != plan['plan_fingerprint']
                        or not check['within_remaining_budget']):
                    raise SummaryInputError('원문 처리 계획 또는 남은 호출 한도가 바뀌었습니다.')
                before = runner.budget_usage()['calls_used_or_reserved']

                def changed():
                    self._publish(identifier, **progress(runner, did))

                runner.on_progress = changed
                self._publish(identifier, status='running', **progress(runner, did))
                runner.run([did], allow_api=True)
                raw = runner.result(did)
                document = env['registry'].documents_by_id[did]
                chunks = env['index'].get_document_chunks(did)
                derived = rebuild_document_summary(raw, document, chunks, env['registry'])
                save_content_result(runner.folder, raw, derived)
                complete = (raw['execution_status'] in COMPLETE
                            and raw['checked_chunk_coverage'] == 1.0)
                diagnostics = [outcome['error_diagnostic'] for outcome in raw['batch_outcomes']
                               if outcome.get('error_diagnostic')]
                self._publish(identifier, status='completed' if complete else 'needs_attention',
                              execution_status=raw['execution_status'],
                              new_api_calls=runner.budget_usage()['calls_used_or_reserved'] - before,
                              error_diagnostics=diagnostics,
                              checked_chunk_coverage=raw['checked_chunk_coverage'],
                              **progress(runner, did))
        except Exception as error:
            self._publish(identifier, status='error', error_diagnostic=safe_llm_diagnostic(error))

    def report(self):
        with self._lock:
            return {'budget_scope': 'Separate persistent summary runs; excluded from question usage.',
                    'jobs': copy.deepcopy(list(self._jobs.values())),
                    'semantic_quality_verified': False}
