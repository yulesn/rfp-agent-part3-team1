"""Pilot invariants, budgets and feedback using synthetic RFPs only."""
import copy
import contextlib
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rfp_rag.retrieval.retrieval import LocalSearchIndex
from regression.conversation.test_session import fixture
from regression.evaluation.test_evaluation import Client
from rfp_rag.pilot.pilot import PilotSession, digest, latest_review, read_record, readiness, review_template, save_review, session_report, session_diagnostics, write_new
from rfp_rag.pilot.web import create_server


def environment(root, *, max_calls=2, cache=False):
    docs, chunks, registry = fixture()
    index = LocalSearchIndex(root / 'index')
    index.build(chunks, {'min_df': 1})
    (root / 'configs').mkdir(exist_ok=True)
    config6 = {'conversation': {}, 'retrieval': {'method': 'rrf', 'top_k': 8,
        'per_doc_top_k': 6, 'fetch_k': 40, 'mmr_lambda': .75, 'max_context_bytes': 42000,
        'max_question_chars': 4000}}
    config8 = {'generation': {'max_calls': max_calls, 'max_total_input_bytes': 600000},
        'pricing': None, 'summary_run': 'absent_summary_run'}
    (root / 'configs/phase6.json').write_text(json.dumps(config6), encoding='utf-8')
    (root / 'configs/phase8.json').write_text(json.dumps(config8), encoding='utf-8')
    return {'root': root, 'documents': docs, 'chunks': chunks, 'registry': registry,
        'index': index, 'phase5_config': {'generation': {'cache_enabled': cache}}}


class PilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.env = environment(Path(temporary.name))
        self.addCleanup(lambda: self.env['index'].close())
        self.client = Client()
        self.pilot = PilotSession(self.env, client=self.client)
        self.pilot.select(['A'])
        self.keys = patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-not-a-real-key'})
        self.keys.start()
        self.addCleanup(self.keys.stop)

    def enable(self, ids=('A',)):
        self.pilot.configure_generation(enabled=True, approved_doc_ids=ids)

    def ask(self, question='사업 예산은?', **kwargs):
        return self.pilot.submit(question, **kwargs)

    def turn_folder(self, record):
        return self.pilot.folder / 'turns' / record['turn_id']

    def test_offline_readiness_and_submit_never_load_key_or_generate(self):
        before = self.env['index'].get_metadata()
        with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')), \
                patch.object(self.pilot.chat.llm, 'generate', side_effect=AssertionError('generation forbidden')):
            self.assertEqual(readiness(self.env)['api_calls'], 0)
            record = self.ask()
            self.assertFalse(record['reviewable'])
            self.assertEqual(record['result']['execution_status'], 'disabled')
            self.assertEqual(self.pilot.saved_summary('A')['status'], 'unavailable')
        self.assertEqual(self.client.bodies, [])
        self.assertEqual(self.env['index'].get_metadata(), before)

    def test_claim_hints_reach_record_review_and_report_without_extra_calls_or_approval(self):
        from regression.pilot.test_claim_audit import NamedItemClient
        from regression.conversation.test_focus import synthetic_environment
        root = self.env['root'] / 'scope_fixture'
        root.mkdir()
        env = synthetic_environment(root)
        self.addCleanup(env['index'].close)
        client = NamedItemClient()
        pilot = PilotSession(env, client=client)
        pilot.configure_generation(enabled=True, approved_doc_ids=['A'])
        pilot.select(['A'])
        record = pilot.submit('구체적인 구축 대상 업무 범위는?')
        self.assertEqual(record['result']['execution_status'], 'completed')
        audit = record['result']['claim_citation_audit']
        self.assertEqual(audit['flagged_claim_indices'], [0])
        self.assertFalse(audit['automatic_semantic_pass'])
        self.assertEqual(review_template(record)['claim_citation_review']['flagged_claim_indices'], [0])
        report = session_report(pilot.folder)
        self.assertEqual(report['claim_citation_review']['claims_with_literal_anchor_gaps'], 1)
        self.assertEqual(report['human_review_status'], 'review_pending')
        self.assertEqual(report['execution_status'], 'records_complete')
        self.assertEqual(len(client.bodies), 1)
        self.assertEqual(report['usage']['live_api_calls'], 0)
        self.assertNotIn('claim_citation_audit', json.dumps(client.bodies))

    def test_old_saved_record_gets_read_only_claim_hints_without_generation_or_reviews(self):
        from regression.pilot.test_claim_audit import NamedItemClient
        from rfp_rag.pilot.pilot import digest, session_diagnostics
        from regression.conversation.test_focus import synthetic_environment
        root = self.env['root'] / 'scope_fixture'
        root.mkdir()
        env = synthetic_environment(root)
        self.addCleanup(env['index'].close)
        pilot = PilotSession(env, client=NamedItemClient())
        pilot.configure_generation(enabled=True, approved_doc_ids=['A'])
        pilot.select(['A'])
        record = pilot.submit('구체적인 구축 범위는?')
        turn = pilot.folder / 'turns' / record['turn_id']
        record['result'].pop('claim_citation_audit')
        record['fingerprint'] = digest({k: v for k, v in record.items() if k != 'fingerprint'})
        (turn / 'record.json').write_text(json.dumps(record))
        (turn / 'human_review.json').write_text('unfinished review edit')
        before = {p: p.read_bytes() for p in pilot.folder.rglob('*') if p.is_file()}
        with patch('rfp_rag.generation.llm.OpenAIGroundedLLM.generate', side_effect=AssertionError('no API')), \
                patch('socket.socket.connect', side_effect=AssertionError('no network')):
            diagnostic = session_diagnostics(pilot.folder)
        self.assertEqual(diagnostic['diagnostic_api_calls'], 0)
        self.assertEqual(diagnostic['turns'][0]['claim_citation_audit']['flagged_claim_indices'], [0])
        self.assertEqual(before, {p: p.read_bytes() for p in pilot.folder.rglob('*') if p.is_file()})

    def test_missing_or_unknown_live_scope_is_rejected(self):
        for ids in [[], ['UNKNOWN']]:
            with self.assertRaises(ValueError):
                self.enable(ids)
        self.assertFalse(self.pilot.chat.llm.config.enabled)
        self.enable(['COPY_A'])
        self.assertEqual(self.pilot.approved_doc_ids, ['A'])

    def test_mode_changes_and_reset_retain_budget_and_client(self):
        self.enable()
        original = self.pilot.chat.llm
        self.ask()
        self.pilot.reset_conversation()
        self.pilot.select(['A'])
        self.pilot.configure_generation(enabled=False)
        self.ask('과업 기간은?')
        self.enable()
        self.ask('주요 사업기간은?')
        blocked = self.ask('사업비의 부가세 조건은?')
        self.assertIs(self.pilot.chat.llm, original)
        self.assertEqual(len(self.client.bodies), 2)
        self.assertEqual(blocked['result']['execution_status'], 'error')
        self.assertTrue(self.pilot.halted)

    def test_comparison_preflight_respects_remaining_session_budget(self):
        self.enable(['A', 'B'])
        self.ask()  # One call left, but two document requests are required.
        self.pilot.select(['A', 'B'])
        record = self.ask('선택한 두 사업의 예산 비교')
        self.assertEqual(record['result']['execution_status'], 'error')
        self.assertEqual(len(self.client.bodies), 1)

    def test_unapproved_document_is_blocked_before_provider_call(self):
        self.enable()
        self.pilot.select(['B'])
        result = self.ask()
        self.assertEqual(result['result']['execution_status'], 'not_approved')
        self.assertEqual(self.client.bodies, [])

    def test_duplicate_submission_returns_same_record_without_another_call(self):
        self.enable()
        first = self.ask(request_id='same-click')
        second = self.ask(request_id='same-click')
        self.assertEqual(first, second)
        self.assertEqual(len(self.client.bodies), 1)
        self.assertEqual(len(list((self.pilot.folder / 'turns').iterdir())), 1)
        with self.assertRaises(ValueError):
            self.ask('기간은?', request_id='same-click')
        self.pilot.configure_generation(enabled=False)
        with self.assertRaises(ValueError):
            self.ask(request_id='same-click')

    def test_failed_scope_filter_change_is_atomic(self):
        before = self.pilot.chat.state
        with self.assertRaises(ValueError):
            self.pilot.select(['A'], filters={'agency': '다른 가상기관'})
        self.assertEqual(self.pilot.chat.state, before)

    def test_new_readonly_index_can_be_attached_without_reusing_closed_connection(self):
        self.env['index'].close()
        fresh = dict(self.env, index=LocalSearchIndex(self.env['root'] / 'index'))
        self.addCleanup(fresh['index'].close)
        self.pilot.attach(fresh)
        self.assertEqual(self.ask()['result']['execution_status'], 'disabled')

    def test_changed_source_snapshot_is_rejected(self):
        other = dict(self.env)
        other['registry'] = copy.copy(self.env['registry'])
        other['registry'].source_hashes = dict(self.env['registry'].source_hashes, A='0' * 64)
        with self.assertRaises(ValueError):
            self.pilot.attach(other)

    def test_request_storage_failure_prevents_api_call(self):
        self.enable()
        with patch('rfp_rag.pilot.pilot.write_new', side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(OSError):
                self.ask()
        self.assertEqual(self.client.bodies, [])
        self.assertEqual(session_report(self.pilot.folder)['execution_status'], 'needs_attention')

    def test_provider_failure_halts_without_automatic_retry_or_body_leak(self):
        self.client.fail = True
        self.enable()
        record = self.ask()
        self.assertEqual(record['usage_delta']['failed_attempts'], 1)
        self.assertTrue(self.pilot.halted)
        with self.assertRaises(ValueError):
            self.ask('다시 요청')
        self.assertEqual(len(self.client.bodies), 1)
        saved = (self.turn_folder(record) / 'record.json').read_text(encoding='utf-8')
        self.assertNotIn('PRIVATE_PROVIDER_BODY', saved)
        self.assertNotIn('synthetic-not-a-real-key', saved)
        self.pilot.resume_after_error()
        self.assertEqual(self.pilot.chat.llm.usage_report()['failed_attempts'], 1)



    def test_feedback_requires_exact_result_and_record_integrity(self):
        self.enable()
        record = self.ask()
        folder = self.turn_folder(record)
        review = review_template(record)
        review['result_fingerprint'] = 'another-result'
        with self.assertRaises(ValueError):
            save_review(folder, review)
        corrupted = copy.deepcopy(record)
        corrupted['result']['markdown'] = 'Changed after inspection'
        (folder / 'record.json').write_text(json.dumps(corrupted), encoding='utf-8')
        with self.assertRaises(ValueError):
            read_record(folder)

    def test_offline_candidates_cannot_be_scored_as_generated_answer(self):
        record = self.ask()
        with self.assertRaises(ValueError):
            save_review(self.turn_folder(record), review_template(record))
        self.assertIsNone(session_report(self.pilot.folder)['accuracy_on_reviewed_live_turns'])

    def test_named_complete_human_review_and_revisions_without_live_accuracy_for_mocks(self):
        self.enable()
        record = self.ask()
        folder = self.turn_folder(record)
        review = review_template(record)
        review['judgements'] = dict.fromkeys(review['judgements'], True)
        save_review(folder, review)
        self.assertEqual(session_report(self.pilot.folder)['human_review_status'], 'review_pending')
        review['reviewer'] = 'Synthetic reviewer'
        save_review(folder, review)
        report = session_report(self.pilot.folder)
        self.assertEqual(report['human_review_status'], 'review_complete')
        self.assertEqual(report['reviewed_turns'], 1)
        self.assertIsNone(report['accuracy_on_reviewed_live_turns'])
        self.assertFalse(report['independent_quality_verified'])
        review['judgements']['conditions_preserved'] = False
        save_review(folder, review)
        self.assertEqual(len(list((folder / 'feedback').glob('*.json'))), 3)
        self.assertEqual(session_report(self.pilot.folder)['fully_correct_reviewed_turns'], 0)
        self.assertTrue(latest_review(folder, record)[1])

    def test_report_reads_only_chosen_session_and_keeps_unreviewed_pending(self):
        self.enable()
        self.ask()
        empty = PilotSession(self.env)
        self.assertEqual(session_report(empty.folder)['recorded_turns'], 0)
        report = session_report(self.pilot.folder)
        self.assertEqual(report['recorded_turns'], 1)
        self.assertEqual(report['human_review_status'], 'review_pending')
        self.assertEqual(report['usage']['mock_calls'], 1)
        self.assertEqual(report['latency']['mock']['n'], 1)

    def test_result_save_failure_retains_unsaved_result_and_stops_new_calls(self):
        self.enable()
        original = write_new
        def fail_record(path, value):
            if Path(path).name == 'record.json':
                raise OSError('synthetic disk failure')
            return original(path, value)
        with patch('rfp_rag.pilot.pilot.write_new', side_effect=fail_record):
            with self.assertRaises(OSError):
                self.ask()
        self.assertEqual(len(self.client.bodies), 1)
        self.assertTrue(self.pilot.halted)
        self.assertEqual(self.pilot.unsaved_record['usage_delta']['mock_calls'], 1)
        self.assertEqual(len(session_report(self.pilot.folder)['incomplete_turns']), 1)
        self.pilot.resume_after_error()
        self.assertFalse(hasattr(self.pilot, 'unsaved_record'))
        self.assertEqual(len(self.client.bodies), 1)
        self.assertEqual(session_report(self.pilot.folder)['recorded_turns'], 1)
        self.assertEqual(session_report(self.pilot.folder)['incomplete_turns'], [])

    def test_invalid_pricing_is_rejected_before_any_generation(self):
        path = self.env['root'] / 'configs/phase8.json'
        config = json.loads(path.read_text())
        config['pricing'] = {'model': 'OTHER', 'input_per_million': 1,
            'cached_input_per_million': 1, 'output_per_million': 1}
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            PilotSession(self.env, client=self.client)
        self.assertEqual(self.client.bodies, [])


def serve_fixture(connection):
    """Child-process entrypoint: SQLite and HTTP run on the same owning thread."""
    with tempfile.TemporaryDirectory() as temporary:
        env = environment(Path(temporary))
        pilot = PilotSession(env, client=Client())
        server = create_server(pilot, port=0)
        connection.send(server.server_port)
        try:
            server.serve_forever(poll_interval=.05)
        finally:
            server.server_close()
            env['index'].close()


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent, child = multiprocessing.get_context('spawn').Pipe()
        cls.process = multiprocessing.get_context('spawn').Process(target=serve_fixture, args=(child,), daemon=True)
        cls.process.start()
        if not parent.poll(30):
            cls.process.terminate()
            raise RuntimeError('synthetic pilot server did not start')
        cls.port = parent.recv()
        cls.base = f'http://127.0.0.1:{cls.port}'

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.join(5)

    def request(self, path, body=None, **headers):
        if body is not None:
            data = json.dumps(body).encode()
            headers.setdefault('Content-Type', 'application/json')
        else:
            data = None
        with urlopen(Request(self.base + path, data=data, headers=headers), timeout=20) as response:
            raw = response.read().decode()
            return raw if path == '/' else json.loads(raw)

    def test_ui_get_and_refresh_do_not_submit_questions(self):
        self.assertIn('입찰메이트', self.request('/'))
        before = self.request('/api/state')
        after = self.request('/api/state')
        self.assertEqual(before['usage'], after['usage'])
        self.assertEqual(before['turns'], after['turns'])

    def test_question_records_reopen_after_refresh_and_reset_keeps_usage(self):
        self.request('/api/select', {'doc_ids': ['A'], 'filters': {}})
        record = self.request('/api/ask', {'question': '사업 예산은?', 'request_id': 'web-test-one'})
        self.assertEqual(record['usage_delta']['api_calls'], 0)
        again = self.request('/api/record?turn_id=' + record['turn_id'])
        self.assertEqual(record, again)
        self.request('/api/reset', {})
        self.assertEqual(self.request('/api/state')['usage']['api_calls'], 0)

    def test_foreign_origin_and_non_json_mutations_are_rejected(self):
        with self.assertRaises(HTTPError) as error:
            self.request('/api/reset', {}, Origin='https://example.invalid')
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error:
            self.request('/api/reset', {}, **{'Content-Type': 'text/plain'})
        self.assertEqual(error.exception.code, 400)

    def test_live_cannot_be_enabled_without_selected_allowlist(self):
        with self.assertRaises(HTTPError) as error:
            self.request('/api/generation', {'enabled': True, 'approved_doc_ids': []})
        self.assertEqual(error.exception.code, 400)
        self.assertFalse(self.request('/api/state')['live_enabled'])


if __name__ == '__main__':
    unittest.main()
