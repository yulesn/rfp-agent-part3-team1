"""End-to-end local summary requests with invented sources and no paid API."""
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rfp_rag.pilot.pilot import PilotSession
from rfp_rag.pilot.summaries import SummaryInputError, WebSummaries
from rfp_rag.pilot.web import create_server
from rfp_rag.retrieval.retrieval import LocalSearchIndex
from rfp_rag.workspace import build_demo_workspace, build_workspace, open_workspace


class SummaryClient:
    def __init__(self, *, fail=False, bad_alias=False, paused=False):
        self.responses = self
        self.bodies = []
        self.fail, self.bad_alias = fail, bad_alias
        self.entered = threading.Event()
        self.release = threading.Event()
        if not paused:
            self.release.set()

    def create(self, **body):
        self.bodies.append(copy.deepcopy(body))
        self.entered.set()
        if not self.release.wait(15):
            raise RuntimeError('synthetic client timed out')
        if self.fail:
            raise RuntimeError('PRIVATE_PROVIDER_BODY_MUST_NOT_APPEAR')
        request = json.loads(body['input'][1]['content'])
        unit = next(row for row in request['evidence'] if row['citable'])
        payload = {'status': 'answered', 'claims': [{
            'doc_id': 'D001', 'category': 'requirements',
            'text': '[scope_requirements|fact] ' + unit['text'].strip(),
            'citations': [{'chunk_id': 'E999999' if self.bad_alias else unit['chunk_id'],
                           'quote': 'SOURCE'}]}], 'missing_information': []}
        return SimpleNamespace(status='completed', output=[], output_text=json.dumps(payload),
                               usage=SimpleNamespace(input_tokens=21, output_tokens=9, total_tokens=30))


def wait_for_summary(service, identifier):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        value = service.status(identifier)
        if value['status'] not in {'queued', 'running'}:
            return value
        time.sleep(.01)
    raise AssertionError('summary worker did not finish')


class SummaryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'workspace'
        build_demo_workspace(self.root)
        self.client = SummaryClient()
        self.env = open_workspace(self.root)
        self.addCleanup(self.env['index'].close)
        network = patch('socket.socket.connect', side_effect=AssertionError('network prohibited'))
        network.start()
        self.addCleanup(network.stop)
        key = patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-test-key'})
        key.start()
        self.addCleanup(key.stop)
        self.service = WebSummaries(self.root, client=self.client)

    def generate(self, did='DEMO_A'):
        plan = self.service.prepare(self.env, did)
        job = self.service.start(self.env, did, plan_id=plan['plan_id'], allow_api=True)
        return plan, wait_for_summary(self.service, job['job_id'])

    def test_planning_and_reading_are_offline_and_need_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            plan = self.service.prepare(self.env, 'DEMO_A')
            saved = self.service.saved(self.env, 'DEMO_A')
        self.assertEqual(plan['api_calls_by_planning'], 0)
        self.assertGreater(plan['new_api_calls'], 0)
        self.assertEqual(saved['status'], 'unavailable')
        self.assertFalse(self.client.bodies)

    def test_generate_full_summary_and_reopen_after_restart_without_paid_calls(self):
        with patch.object(LocalSearchIndex, 'search', side_effect=AssertionError('not a top-k summary')):
            plan, job = self.generate()
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['completed_batches'], plan['batch_count'])
        self.assertEqual(job['budget']['live_api_calls'], 0)
        self.assertEqual(job['budget']['mock_calls'], plan['new_api_calls'])
        saved = self.service.saved(self.env, 'DEMO_A')
        result = saved['result']
        self.assertEqual(len(result['fields']), 11)
        self.assertEqual(result['checked_chunk_coverage'], 1)
        self.assertEqual(set(result['source_chunk_ids']),
                         {c['chunk_id'] for c in self.env['index'].get_document_chunks('DEMO_A')})
        self.assertFalse(result['semantic_quality_verified'])
        self.assertFalse(result['human_review_completed'])
        self.assertNotIn('rejected_claims', result)
        count = len(self.client.bodies)
        restart = WebSummaries(self.root, client=self.client)
        self.assertEqual(restart.saved(self.env, 'DEMO_A'), saved)
        replay_plan = restart.prepare(self.env, 'DEMO_A')
        replay = restart.start(self.env, 'DEMO_A', plan_id=replay_plan['plan_id'], allow_api=True)
        self.assertEqual(wait_for_summary(restart, replay['job_id'])['new_api_calls'], 0)
        self.assertEqual(len(self.client.bodies), count)
        self.assertEqual(replay_plan['budget']['calls_used_or_reserved'], count)

    def test_citations_are_original_text_in_selected_document_only(self):
        self.generate()
        result = self.service.saved(self.env, 'DEMO_A')['result']
        self.assertTrue(result['claims'])
        chunks = {c['chunk_id']: c for c in self.env['chunks']}
        for claim in result['claims']:
            self.assertEqual(claim['doc_id'], 'DEMO_A')
            for cite in claim['citations']:
                self.assertEqual(chunks[cite['chunk_id']]['doc_id'], 'DEMO_A')
                self.assertIn(cite['quote'], chunks[cite['chunk_id']]['text'])
        wire = json.dumps(self.client.bodies, ensure_ascii=False)
        self.assertNotIn('가상공원공단', wire)

    def test_duplicate_post_returns_same_job_and_does_not_spend_again(self):
        self.client.release.clear()
        self.addCleanup(self.client.release.set)
        plan = self.service.prepare(self.env, 'DEMO_A')
        job = self.service.start(self.env, 'DEMO_A', plan_id=plan['plan_id'], allow_api=True)
        self.assertTrue(self.client.entered.wait(5))
        repeated = self.service.start(self.env, 'DEMO_A', plan_id=plan['plan_id'], allow_api=True)
        self.assertEqual(repeated['job_id'], job['job_id'])
        self.assertEqual(len(self.client.bodies), 1)
        self.assertEqual(repeated['budget']['calls_used_or_reserved'], 1)
        with self.assertRaises(SummaryInputError):
            self.service.prepare(self.env, 'DEMO_B')
        self.client.release.set()
        self.assertEqual(wait_for_summary(self.service, job['job_id'])['status'], 'completed')
        again = self.service.start(self.env, 'DEMO_A', plan_id=plan['plan_id'], allow_api=True)
        self.assertEqual(again['job_id'], job['job_id'])
        self.assertEqual(len(self.client.bodies), 1)

    def test_api_permission_and_source_bound_plan_are_required(self):
        plan = self.service.prepare(self.env, 'DEMO_A')
        for value in (False, 'true', 1, None):
            with self.assertRaises(SummaryInputError):
                self.service.start(self.env, 'DEMO_A', plan_id=plan['plan_id'], allow_api=value)
        for did, identifier in [('DEMO_B', plan['plan_id']), ('DEMO_A', 'unknown')]:
            with self.assertRaises(SummaryInputError):
                self.service.start(self.env, did, plan_id=identifier, allow_api=True)
        self.assertFalse(self.client.bodies)

    def test_provider_failure_is_incomplete_and_does_not_retry_after_restart(self):
        self.client.fail = True
        _, job = self.generate()
        self.assertEqual(job['status'], 'needs_attention')
        self.assertEqual(job['failed_batches'], 1)
        self.assertLess(job['checked_chunk_coverage'], 1)
        self.assertNotIn('PRIVATE_PROVIDER_BODY', json.dumps(job))
        restart = WebSummaries(self.root, client=self.client)
        plan = restart.prepare(self.env, 'DEMO_A')
        self.assertFalse(plan['can_start'])
        with self.assertRaises(SummaryInputError):
            restart.start(self.env, 'DEMO_A', plan_id=plan['plan_id'], allow_api=True)
        self.assertEqual(len(self.client.bodies), 1)

    def test_invalid_citation_cannot_be_displayed_as_completed_summary(self):
        self.client.bad_alias = True
        _, job = self.generate()
        self.assertEqual(job['status'], 'needs_attention')
        self.assertEqual(job['checked_chunk_coverage'], 0)
        saved = self.service.saved(self.env, 'DEMO_A')
        self.assertNotEqual(saved['result']['execution_status'], 'completed')
        for claim in saved['result']['claims']:
            self.assertTrue(claim['origin'].startswith('local_source_'))

    def test_summary_has_a_separate_budget_and_does_not_modify_chat(self):
        pilot = PilotSession(self.env)
        before = pilot.chat.llm.usage_report()
        self.generate()
        self.assertEqual(pilot.chat.llm.usage_report(), before)
        self.assertEqual(pilot.records, [])
        self.assertTrue(self.service.report()['jobs'])

    def test_unknown_document_and_path_inputs_are_rejected(self):
        for value in ('missing', '../DEMO_A', '', None):
            with self.assertRaises((ValueError, TypeError)):
                self.service.prepare(self.env, value)
        with self.assertRaises(SummaryInputError):
            self.service.status('../../secret')
        self.assertFalse(self.client.bodies)

    def test_new_document_and_last_batch_are_included_without_phase5_artifacts(self):
        source = self.base / 'new-document.jsonl'
        doc = copy.deepcopy(self.env['documents'][0])
        doc.update(doc_id='NEW_RFP_2031', canonical_doc_id='NEW_RFP_2031',
                   file_name='새문서_샘플.pdf', sha256='8' * 64)
        doc['blocks'] = [{'block_id': f'p{i}', 'page': i, 'kind': 'paragraph',
                          'locator': f'PDF p.{i}',
                          'text': f'SFR-{i:03d} 신규 기능 {i}: ' + '과업 범위를 조건과 함께 보존합니다. ' * 15}
                         for i in range(1, 31)]
        doc['blocks'][-1]['text'] += ' LAST_PAGE_SENTINEL_2031'
        source.write_text(json.dumps(doc, ensure_ascii=False) + '\n', encoding='utf-8')
        config = self.base / 'long.yaml'
        config.write_text('summary:\n  max_batch_bytes: 15000\n', encoding='utf-8')
        target = self.base / 'new-workspace'
        build_workspace(source, target, config=config)
        service = WebSummaries(target, client=self.client)
        with open_workspace(target) as env:
            plan = service.prepare(env, doc['doc_id'])
            self.assertTrue(plan['can_start'], plan)
            self.assertGreater(plan['batch_count'], 1)
            job = service.start(env, doc['doc_id'], plan_id=plan['plan_id'], allow_api=True)
            final = wait_for_summary(service, job['job_id'])
            self.assertEqual(final['status'], 'completed', final)
            saved = service.saved(env, doc['doc_id'])
            self.assertEqual(saved['result']['checked_chunk_coverage'], 1)
        self.assertIn('LAST_PAGE_SENTINEL_2031', json.dumps(self.client.bodies))
        self.assertEqual(len(self.client.bodies), plan['new_api_calls'])

    def test_oversized_document_is_blocked_before_any_call(self):
        config = self.base / 'small-budget.yaml'
        config.write_text('generation:\n  max_total_input_bytes: 60000\n', encoding='utf-8')
        target = self.base / 'limited'
        source = self.base / 'large-source.jsonl'
        document = copy.deepcopy(self.env['documents'][0])
        document['blocks'][0]['text'] *= 200
        source.write_text(json.dumps(document, ensure_ascii=False) + '\n', encoding='utf-8')
        build_workspace(source, target, config=config)
        with open_workspace(target) as env:
            service = WebSummaries(target, client=self.client)
            plan = service.prepare(env, 'DEMO_A')
            self.assertFalse(plan['can_start'])
            self.assertFalse(plan['within_budget'])
            with self.assertRaises(SummaryInputError):
                service.start(env, 'DEMO_A', plan_id=plan['plan_id'], allow_api=True)
        self.assertFalse(self.client.bodies)


def serve_summary_fixture(connection):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / 'web'
        build_demo_workspace(root)
        with open_workspace(root) as env:
            client = SummaryClient(paused=True)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-test-key'}):
                server = create_server(PilotSession(env), port=0, summary_client=client)
                connection.send(server.server_port)

                def release():
                    # Keep the request in flight long enough to test responsive GETs.
                    client.entered.wait(20)
                    client.release.wait(.5)
                    client.release.set()

                threading.Thread(target=release, daemon=True).start()
                try:
                    server.serve_forever(poll_interval=.01)
                finally:
                    server.server_close()


class WebSummaryHTTPTests(unittest.TestCase):
    def setUp(self):
        parent, child = multiprocessing.get_context('spawn').Pipe()
        self.process = multiprocessing.get_context('spawn').Process(
            target=serve_summary_fixture, args=(child,), daemon=True)
        self.process.start()
        self.addCleanup(self.stop)
        if not parent.poll(20):
            raise AssertionError('web server did not start')
        self.base = f'http://127.0.0.1:{parent.recv()}'

    def stop(self):
        self.process.terminate()
        self.process.join(5)

    def request(self, path, body=None, **headers):
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers.setdefault('Content-Type', 'application/json')
        with urlopen(Request(self.base + path, data=data, headers=headers), timeout=10) as response:
            return json.load(response)

    def test_explicit_generation_with_progress_refresh_and_saved_reopen(self):
        self.request('/api/select', {'doc_ids': ['DEMO_A']})
        plan = self.request('/api/summary/plan', {'doc_id': 'DEMO_A'})
        self.assertEqual(self.request('/api/summary?doc_id=DEMO_A')['status'], 'unavailable')
        with self.assertRaises(HTTPError):
            self.request('/api/summary/generate', {'doc_id': 'DEMO_A', 'plan_id': plan['plan_id'], 'allow_api': True})
        self.request('/api/generation', {'enabled': True, 'approved_doc_ids': ['DEMO_A']})
        request = {'doc_id': 'DEMO_A', 'plan_id': plan['plan_id'], 'allow_api': True}
        job = self.request('/api/summary/generate', request)
        self.assertEqual(self.request('/api/summary/generate', request)['job_id'], job['job_id'])
        state = self.request('/api/state')
        self.assertEqual(state['summary_job']['job_id'], job['job_id'])
        self.assertEqual(state['usage']['api_calls'], 0)
        self.assertEqual(state['turns'], [])
        with self.assertRaises(HTTPError):
            self.request('/api/select', {'doc_ids': ['DEMO_B']})
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            final = self.request('/api/summary/status?job_id=' + job['job_id'])
            if final['status'] not in {'queued', 'running'}:
                break
            time.sleep(.02)
        self.assertEqual(final['status'], 'completed', final)
        saved = self.request('/api/summary?doc_id=DEMO_A')
        self.assertEqual(saved['result']['checked_chunk_coverage'], 1)
        self.assertEqual(self.request('/api/summary?doc_id=DEMO_A'), saved)
        self.assertEqual(self.request('/api/report')['web_summary']['jobs'][0]['usage']['mock_calls'], 1)

    def test_wrong_scope_unapproved_doc_and_foreign_origin_are_rejected(self):
        self.request('/api/select', {'doc_ids': ['DEMO_A', 'DEMO_B']})
        with self.assertRaises(HTTPError):
            self.request('/api/summary/plan', {'doc_id': 'DEMO_A'})
        self.request('/api/select', {'doc_ids': ['DEMO_A']})
        plan = self.request('/api/summary/plan', {'doc_id': 'DEMO_A'})
        self.request('/api/generation', {'enabled': True, 'approved_doc_ids': ['DEMO_B']})
        request = {'doc_id': 'DEMO_A', 'plan_id': plan['plan_id'], 'allow_api': True}
        with self.assertRaises(HTTPError):
            self.request('/api/summary/generate', request)
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/summary/generate', request, Origin='https://example.invalid')
        self.assertEqual(caught.exception.code, 403)
        self.assertIsNone(self.request('/api/state')['summary_job'])


if __name__ == '__main__':
    unittest.main()
