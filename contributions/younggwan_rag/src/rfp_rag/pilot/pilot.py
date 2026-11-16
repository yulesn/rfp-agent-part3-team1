"""Keep conversation and usage across UI operations, without sharing SQLite handles.

Only submit() may ask the backend. Listing, rerendering, feedback, summaries,
readiness and reset never generate an answer. Every explicit submission gets
an immutable result and blank review bound to its exact bytes.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from rfp_rag.generation.costs import estimate_usage_cost
from rfp_rag.generation.llm import safe_llm_diagnostic
from rfp_rag.conversation.runtime import GroundedBackend, make_conversation
from rfp_rag.conversation.session import ConversationSession
from rfp_rag.evaluation.evaluate import _load_summary
from rfp_rag.evaluation.review import JUDGEMENTS
from . import VERSION
from .claim_audit import claim_citation_audit, claim_review_summary


COUNTERS = ('api_calls', 'live_api_calls', 'mock_calls', 'cache_hits',
    'failed_attempts', 'attempted_input_bytes', 'reported_input_tokens',
    'reported_output_tokens', 'reported_cached_input_tokens',
    'reported_total_tokens', 'usage_unknown_attempts', 'request_latency_ms')
FAILED = {'error', 'validation_failed', 'incomplete', 'not_approved'}


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    """Exclusive writes; existing records are never silently overwritten."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8]


def readiness(env):
    """Report the already validated local snapshot; no new index or API calls."""
    catalog = env['index'].get_catalog()
    return {'version': VERSION, 'status': 'local_snapshot_ready',
        'canonical_documents': len(catalog), 'original_document_ids': len(env['documents']),
        'chunks': len(env['chunks']), 'source_fingerprint': digest(env['registry'].source_hashes),
        'index_fingerprint': env['index'].get_metadata()['index_fingerprint'],
        'environment_compatible': env.get('environment', {}).get('compatible'),
        'default_live_enabled': False, 'api_calls': 0,
        'independent_quality_verified': False,
        'phase7_development_baseline': {'source_spans': [130, 141], 'scope_turns': [71, 71],
            'remaining_source_spans': 11, 'development_summary_cases_without_saved_revision': 9,
            'is_current_pilot_measurement': False},
        'pilot_pending': ['실제 사용자 질문과 원문 대조', '새 문서·질문의 독립 표본 확보',
                          '실제 API 지연·사용량 측정']}


def embedding_usage(env):
    """Query embeddings are accounted separately from grounded generation."""
    report = getattr(env['index'], 'usage_report', None)
    return report() if report else None


def read_record(folder):
    record = load_json(Path(folder) / 'record.json')
    payload = {k: v for k, v in record.items() if k != 'fingerprint'}
    if record.get('version') != VERSION or record.get('fingerprint') != digest(payload):
        raise ValueError('실행 기록의 지문이 다릅니다.')
    return record


def review_template(record):
    return {'version': VERSION, 'turn_id': record['turn_id'],
        'result_fingerprint': record['fingerprint'], 'reviewer': '',
        'judgements': dict.fromkeys(JUDGEMENTS), 'notes': '',
        'automatic_semantic_pass': False,
        'claim_citation_review': {'flagged_claim_indices':
            claim_citation_audit(record['result'])['flagged_claim_indices'],
            'instruction': '각 문장 전체가 그 문장에 붙은 인용으로 뒷받침되는지 확인하세요. '
                '명칭 일치 신호가 있어도 조건·의미·누락은 별도로 검토해야 합니다.'}}


def _check_review(record, review):
    if (review.get('version') != VERSION or review.get('turn_id') != record['turn_id']
            or review.get('result_fingerprint') != record['fingerprint']):
        raise ValueError('다른 답변에 대한 검토입니다.')
    judgements = review.get('judgements')
    if (not isinstance(judgements, dict) or set(judgements) != set(JUDGEMENTS)
            or any(v is not None and type(v) is not bool for v in judgements.values())
            or not isinstance(review.get('notes'), str)
            or not isinstance(review.get('reviewer'), str)):
        raise ValueError('검토자와 true/false/null 판정을 확인하세요.')
    return bool(review['reviewer'].strip()) and all(v is not None for v in judgements.values())


def save_review(turn_folder, review):
    """Accept only explicit, correctly bound human input; keep older revisions."""
    folder = Path(turn_folder)
    record = read_record(folder)
    if not record['reviewable']:
        raise ValueError('생성되지 않은 검색 후보에는 답변 정답 판정을 기록하지 않습니다.')
    _check_review(record, review)
    destination = folder / 'feedback'
    destination.mkdir(exist_ok=True)
    path = destination / (stamp() + '.json')
    write_new(path, review)
    return path


def latest_review(turn_folder, record):
    folder = Path(turn_folder)
    revisions = sorted((folder / 'feedback').glob('*.json'))
    review = load_json(revisions[-1] if revisions else folder / 'human_review.json')
    complete = _check_review(record, review)
    return review, complete


class PilotSession:
    """One browser/kernel budget. Reset and mode changes retain this LLM object.

    attach() replaces the backend's environment with a newly opened read-only
    index on the current UI thread. No cached SQLite connection is reused by
    another UI thread or browser session. The caller closes env['index'].
    """
    def __init__(self, env, *, client=None):
        self.root = Path(env['root']).resolve()
        self.settings = load_json(self.root / 'configs/phase8.json')
        self.phase6 = load_json(self.root / 'configs/phase6.json')
        self.source_fingerprint = digest(env['registry'].source_hashes)
        self.index_fingerprint = env['index'].get_metadata()['index_fingerprint']
        self.approved_doc_ids = []
        generation = dict(env['phase5_config']['generation'])
        generation.update(self.settings['generation'])
        self.chat = make_conversation(env, client=client, enabled=False,
            config=self.phase6['conversation'], retrieval=self.phase6['retrieval'], generation=generation)
        if self.settings.get('pricing') is not None:
            estimate_usage_cost({}, self.settings['pricing'])
            if self.settings['pricing'].get('model') != self.chat.llm.config.model:
                raise ValueError('단가의 모델명과 생성 모델명이 다릅니다.')
        self.session_id = stamp()
        self.folder = self.root / 'phase8_outputs/sessions' / self.session_id
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.folder / 'turns').mkdir()
        self.records = []
        self._requests = {}
        self.halted = False
        self.attach(env)
        write_new(self.folder / 'session.json', {'version': VERSION, 'session_id': self.session_id,
            'source_fingerprint': self.source_fingerprint, 'index_fingerprint': self.index_fingerprint,
            'generation_limits': self.settings['generation'], 'independent_quality_verified': False,
            'budget_scope': 'One PilotSession; reset/mode changes do not reset usage. A new process/browser session starts a new budget.'})

    def attach(self, env):
        if (Path(env['root']).resolve() != self.root
                or digest(env['registry'].source_hashes) != self.source_fingerprint
                or env['index'].get_metadata()['index_fingerprint'] != self.index_fingerprint):
            raise ValueError('원문·인덱스가 바뀌었습니다. 새 파일럿 세션을 시작하세요.')
        self.env = env
        self.chat.backend = GroundedBackend(env, self.chat.llm, retrieval=self.phase6['retrieval'],
            approved_doc_ids=self.approved_doc_ids if self.chat.llm.config.enabled else None)

    def configure_generation(self, *, enabled, approved_doc_ids=()):
        if type(enabled) is not bool:
            raise TypeError('enabled는 True/False여야 합니다.')
        approved = self.env['registry'].resolve_doc_ids(list(approved_doc_ids)) if approved_doc_ids else []
        if enabled and not approved:
            raise ValueError('API에 전송할 문서를 선택하세요.')
        self.approved_doc_ids = approved
        # Keep this exact client and all counters, including failed attempts.
        self.chat.llm.allowed_sources = {d: self.env['registry'].source_hashes[d] for d in approved}
        self.chat.llm.config = replace(self.chat.llm.config, enabled=enabled)
        self.attach(self.env)

    def select(self, doc_ids, *, filters=None):
        values = self.chat.filters if filters is None else filters
        # Validate the complete change before mutating the live conversation.
        candidate = ConversationSession(self.env['documents'], config=self.chat.config)
        candidate.set_filters(values)
        candidate.select_documents(doc_ids)
        self.chat.set_filters(values)
        return self.chat.select_documents(doc_ids)

    def reset_conversation(self):
        self.chat.reset()

    def resume_after_error(self):
        if hasattr(self, 'unsaved_record'):
            self.retry_save()
        self.halted = False

    def retry_save(self):
        """Persist an already received response; never resubmit a model request."""
        if not hasattr(self, 'unsaved_record'):
            return
        record = self.unsaved_record
        folder = self.folder / 'turns' / record['turn_id']
        for name, value in [('record.json', record), ('human_review.json', review_template(record))]:
            path = folder / name
            if path.exists():
                try:
                    existing = load_json(path)
                except (ValueError, OSError):
                    path.rename(folder / (name + '.partial_' + stamp()))
                else:
                    if name == 'record.json' and existing != record:
                        raise ValueError('저장된 결과가 메모리의 결과와 다릅니다.')
                    if name == 'human_review.json':
                        _check_review(record, existing)
                    continue
            write_new(path, value)
        self.records.append(record)
        if record['request_id'] is not None:
            self._requests[record['request_id']] = record
        del self.unsaved_record

    def submit(self, question, *, request_id=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('질문을 입력하세요.')
        if len(question) > self.chat.config.max_question_chars:
            raise ValueError('질문 길이가 설정 상한을 넘었습니다.')
        submission_context = {'active_doc_ids': self.chat.active_doc_ids, 'filters': self.chat.filters,
            'approved_doc_ids': list(self.approved_doc_ids), 'model': self.chat.llm.config.model,
            'source_fingerprint': self.source_fingerprint, 'index_fingerprint': self.index_fingerprint}
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
                raise ValueError('제출 식별자를 확인하세요.')
            if request_id in self._requests:
                previous = self._requests[request_id]
                if (previous['question'] != question or previous['live_enabled'] != self.chat.llm.config.enabled
                        or previous.get('submission_context') != submission_context):
                    raise ValueError('문서·필터·API 허용 범위 또는 질문이 바뀌었습니다. 새 제출 식별자를 사용하세요.')
                return copy.deepcopy(previous)
        if self.halted:
            raise ValueError('실패 원인을 확인한 뒤 계속 버튼을 눌러 주세요. 자동 재시도하지 않습니다.')
        turn_id = f'{len(self.records) + 1:04d}_{uuid.uuid4().hex[:8]}'
        folder = self.folder / 'turns' / turn_id
        folder.mkdir(exist_ok=False)
        before = self.chat.llm.usage_report()
        embedding_before = embedding_usage(self.env)
        # Fail before any API call if local storage cannot record the request.
        write_new(folder / 'request.json', {'question': question, 'request_id': request_id,
            'active_doc_ids': self.chat.active_doc_ids, 'filters': self.chat.filters,
            'live_enabled': self.chat.llm.config.enabled, 'approved_doc_ids': self.approved_doc_ids,
            'submission_context': submission_context})
        started = time.perf_counter()
        try:
            result = self.chat.ask(question)
        except Exception as error:
            result = {'status': 'not_generated', 'execution_status': 'error',
                'answer_generated': False, 'claims': [], 'source_chunks': [],
                'markdown': '요청을 완료하지 못했습니다. 실행 진단을 확인하세요.',
                'error_diagnostic': safe_llm_diagnostic(error)}
        return self._record(folder, question, result, before, started, request_id=request_id,
                            submission_context=submission_context, embedding_before=embedding_before)

    def saved_summary(self, doc_id, *, run_folder=None):
        did = self.env['registry'].resolve_doc_ids([doc_id])[0]
        base = Path(run_folder) if run_folder is not None else self.root / self.settings['summary_run']
        result, reason = _load_summary(self.env, base, did)
        if result is None:
            return {'status': 'unavailable', 'reason': reason, 'doc_id': did, 'api_calls': 0}
        return {'status': 'saved_summary', 'result': result, 'doc_id': did, 'api_calls': 0,
            'semantic_quality_verified': False}

    def _record(self, folder, question, result, before, started, *, request_id, submission_context,
                embedding_before=None):
        result = copy.deepcopy(result)
        result['claim_citation_audit'] = claim_citation_audit(result)
        after = self.chat.llm.usage_report()
        delta = {key: after[key] - before[key] for key in COUNTERS}
        origin = ('mock' if delta['mock_calls'] else 'mixed_live_cache' if delta['live_api_calls'] and delta['cache_hits']
            else 'live_api' if delta['live_api_calls'] else 'local_cache' if delta['cache_hits']
            else 'no_model_call')
        record = {'version': VERSION, 'session_id': self.session_id, 'turn_id': folder.name,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'question': question,
            'source_fingerprint': self.source_fingerprint, 'index_fingerprint': self.index_fingerprint,
            'model': self.chat.llm.config.model,
            'request_id': request_id, 'result': copy.deepcopy(result), 'usage_delta': delta,
            'submission_context': copy.deepcopy(submission_context),
            'usage_cumulative': after, 'latency_ms': round((time.perf_counter() - started) * 1000, 3),
            'latency_origin': origin, 'live_enabled': self.chat.llm.config.enabled,
            'execution_origin': after['execution_mode'],
            'reviewable': bool(self.chat.llm.config.enabled and result.get('execution_status') not in FAILED
                and (result.get('answer_generated') or result.get('status') in {'insufficient_evidence', 'needs_clarification'})),
            'novelty': 'unverified', 'independent_quality_verified': False,
            'cost_estimate': estimate_usage_cost(delta, self.settings['pricing']) if self.settings.get('pricing') else None}
        embedding_after = embedding_usage(self.env)
        if embedding_after is not None:
            record['embedding_usage_cumulative'] = embedding_after
            record['embedding_usage_delta'] = {
                key: embedding_after.get(key, 0) - (embedding_before or {}).get(key, 0)
                for key in ('api_calls', 'cache_hits', 'failed_attempts',
                            'reported_input_tokens', 'attempted_input_bytes')}
        record['fingerprint'] = digest(record)
        try:
            write_new(folder / 'record.json', record)
            write_new(folder / 'human_review.json', review_template(record))
        except OSError:
            self.halted = True
            self.unsaved_record = record
            raise
        self.records.append(record)
        if request_id is not None:
            self._requests[request_id] = record
        self.halted = result.get('execution_status') in FAILED
        return copy.deepcopy(record)


def session_report(folder):
    """Read the chosen session only. No newest-run guessing or semantic auto-pass."""
    folder = Path(folder)
    session = load_json(folder / 'session.json')
    records = []
    incomplete = []
    reviews = []
    for turn in sorted((folder / 'turns').iterdir()):
        if not (turn / 'record.json').is_file() or not (turn / 'human_review.json').is_file():
            incomplete.append(turn.name)
            continue
        record = read_record(turn)
        if record['session_id'] != session['session_id']:
            raise ValueError('다른 세션의 실행 기록입니다.')
        review, complete = latest_review(turn, record)
        records.append(record)
        if record['reviewable']:
            reviews.append((record, review, complete))
    completed = [r for r in reviews if r[2]]
    correct = sum(all(r[1]['judgements'].values()) for r in completed)
    real_completed = [r for r in completed if r[0]['execution_origin'] == 'live_openai']
    real_correct = sum(all(r[1]['judgements'].values()) for r in real_completed)
    usage = {key: sum(r['usage_delta'][key] for r in records) for key in COUNTERS}
    embedding_totals = {key: sum(r.get('embedding_usage_delta', {}).get(key, 0) for r in records)
                        for key in ('api_calls', 'cache_hits', 'failed_attempts',
                                    'reported_input_tokens', 'attempted_input_bytes')}
    latencies = {}
    for origin in sorted({r['latency_origin'] for r in records}):
        values = sorted(r['latency_ms'] for r in records if r['latency_origin'] == origin)
        def percentile(p):
            position = (len(values) - 1) * p
            left = int(position)
            return round(values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left), 3)
        latencies[origin] = {'n': len(values), 'p50_ms': percentile(.5), 'p95_ms': percentile(.95)}
    failures = [r['turn_id'] for r in records if r['result'].get('execution_status') in FAILED]
    return {'version': VERSION, 'session_id': session['session_id'], 'recorded_turns': len(records),
        'execution_status': 'needs_attention' if failures or incomplete else 'records_complete',
        'failed_turns': failures, 'failed_turn_diagnostics': [_failure_diagnostic(r) for r in records
            if r['result'].get('execution_status') in FAILED],
        'incomplete_turns': incomplete, 'usage': usage, 'embedding_usage': embedding_totals,
        'latency': latencies,
        'claim_citation_review': claim_review_summary([claim_citation_audit(r['result']) for r in records]),
        'human_review_status': 'review_complete' if reviews and len(completed) == len(reviews) else 'review_pending',
        'reviewable_turns': len(reviews), 'reviewed_turns': len(completed), 'fully_correct_reviewed_turns': correct,
        'accuracy_on_reviewed_live_turns': real_correct / len(real_completed) if real_completed else None,
        'reviewed_live_turns': len(real_completed), 'independent_quality_verified': False,
        'scope': 'Explicitly reviewed local pilot turns only. Question/document novelty is unverified; no unseen-test or corpus-wide accuracy.'}


def _failure_diagnostic(record):
    answer = record['result']
    return {'turn_id': record['turn_id'], 'status': answer.get('status'),
        'execution_status': answer.get('execution_status'),
        'error_diagnostic': copy.deepcopy(answer.get('error_diagnostic')),
        'claim_citation_audit': claim_citation_audit(answer),
        'validation_errors': copy.deepcopy((answer.get('validation') or {}).get('errors', []))}


def session_diagnostics(folder):
    """Inspect one saved pilot session without bootstrap, key, model or reviews."""
    folder = Path(folder)
    session = load_json(folder / 'session.json')
    records = []
    incomplete = []
    for turn in sorted((folder / 'turns').iterdir()):
        if not (turn / 'record.json').is_file():
            incomplete.append(turn.name)
            continue
        record = read_record(turn)
        if record['session_id'] != session['session_id']:
            raise ValueError('다른 세션의 실행 기록입니다.')
        records.append(record)
    return {'version': 'phase8-saved-pilot-diagnostics-v1', 'session_id': session['session_id'],
        'diagnostic_api_calls': 0, 'read_only': True, 'recorded_turns': len(records),
        'incomplete_turns': incomplete, 'turns': [_failure_diagnostic(r) for r in records],
        'semantic_correctness_verified': False}
