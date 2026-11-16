"""Public API, CLI, corpus bindings and repeatable notebooks using invented data."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.cli import main
from rfp_rag.pilot.pilot import PilotSession
from rfp_rag.workspace import build_demo_workspace, build_workspace, open_workspace
from rfp_rag.workflows import search, submit_questions, summarize, validate, validate_once


PACKAGE = Path(__file__).resolve().parents[1]


class PublicWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'corpus'
        build_demo_workspace(self.root)

    def test_build_is_idempotent_and_keeps_original_manifest(self):
        manifest = self.root / 'reports/run_manifest.json'
        before = manifest.read_bytes()
        self.assertTrue(build_demo_workspace(self.root)['reused'])
        self.assertEqual(before, manifest.read_bytes())

    def test_changed_source_requires_new_workspace(self):
        source = self.root.parent / 'different.jsonl'
        source.write_text((self.root/'data/source_documents.jsonl').read_text()+'\n')
        with self.assertRaisesRegex(ValueError, '새 workspace'):
            build_workspace(source, self.root)

    def test_source_tampering_is_rejected_before_search(self):
        with (self.root/'data/source_documents.jsonl').open('a') as f:
            f.write('\n')
        with self.assertRaises(ValueError):
            open_workspace(self.root)

    def test_code_mismatch_is_rejected(self):
        with patch('rfp_rag.workspace.package_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, '구현 코드'):
                open_workspace(self.root)

    def test_unknown_selection_fails_and_search_keeps_selected_document(self):
        with open_workspace(self.root) as env:
            hits = search(env, '사업 예산과 기간은?', doc_ids=['DEMO_A'])
            self.assertTrue(hits)
            self.assertEqual({h['doc_id'] for h in hits}, {'DEMO_A'})
            with self.assertRaises(ValueError):
                search(env, '예산은?', doc_ids=['UNKNOWN'])

    def test_offline_cell_replay_has_no_api_calls_or_duplicate_records(self):
        with open_workspace(self.root) as env:
            pilot = PilotSession(env)
            with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                first = submit_questions(pilot, ['사업예산은?'], doc_ids=['DEMO_A'])
                second = submit_questions(pilot, ['사업예산은?'], doc_ids=['DEMO_A'])
            self.assertEqual(first, second)
            self.assertEqual(len(pilot.records), 1)
            self.assertEqual(pilot.chat.llm.usage_report()['api_calls'], 0)
            self.assertEqual(first[0]['result']['execution_status'], 'disabled')

    def test_live_requires_key_and_explicit_bool(self):
        with open_workspace(self.root) as env:
            pilot = PilotSession(env)
            with self.assertRaises(TypeError):
                submit_questions(pilot, ['예산은?'], doc_ids=['DEMO_A'], live='False')
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
                    submit_questions(pilot, ['예산은?'], doc_ids=['DEMO_A'], live=True)
            self.assertEqual(pilot.chat.llm.usage_report()['api_calls'], 0)

    def test_summary_remains_incomplete_without_generation(self):
        with open_workspace(self.root) as env:
            with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                runner, results = summarize(env, ['DEMO_A'])
            self.assertEqual(results[0]['execution_status'], 'awaiting_api')
            self.assertFalse(results[0].get('human_review_completed', False))
            self.assertTrue((runner.folder/'content_documents/DEMO_A/latest.json').exists())

    def test_all_six_validation_cases_are_offline_and_semantics_remain_pending(self):
        with open_workspace(self.root) as env:
            with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                folder, report = validate(env, ['DEMO_A'])
            self.assertEqual(report['executed_cases'], 6)
            self.assertTrue(report['mechanical_checks_passed'])
            self.assertEqual(report['session']['usage']['live_api_calls'], 0)
            self.assertIsNone(report['semantic_accuracy'])
            self.assertFalse(report['automatic_semantic_pass'])

    def test_new_cli_catalog_ask_summary_and_validation_share_workspace(self):
        for command, extra in [('catalog', []), ('ask', ['--question','사업 예산은?']),
                               ('summarize', []), ('validate', [])]:
            argv = [command, '--workspace', str(self.root)]
            if command != 'catalog':
                argv += ['--doc-id', 'DEMO_A']
            with redirect_stdout(io.StringIO()) as out:
                result = main(argv + extra)
            self.assertEqual(result, 0, command)
            self.assertIsNotNone(json.loads(out.getvalue()))

    def test_validation_notebook_replay_reuses_recorded_session(self):
        runs = {}
        with open_workspace(self.root) as env:
            first, report = validate_once(env, ['DEMO_A'], runs=runs)
            with patch('rfp_rag.workflows.run_validation', side_effect=AssertionError('duplicate validation')):
                second, replay = validate_once(env, ['DEMO_A'], runs=runs)
            self.assertEqual(first, second)
            self.assertEqual(report['session_id'], replay['session_id'])
            self.assertEqual(replay['session']['usage']['api_calls'], 0)

    def test_notebooks_run_all_twice_offline_without_notebook_runtime(self):
        # Cells contain standard Python and call the same tested workflow functions.
        previous = Path.cwd()
        os.chdir(self.root.parent)
        self.addCleanup(os.chdir, previous)
        for path in sorted((PACKAGE/'notebooks').glob('*.ipynb')):
            nb = json.loads(path.read_text())
            namespace = {'__name__': '__main__'}
            with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
                for repeat in range(2):
                    with redirect_stdout(io.StringIO()):
                        for cell in nb['cells']:
                            if cell['cell_type'] == 'code':
                                self.assertIsNone(cell['execution_count'])
                                self.assertEqual(cell['outputs'], [])
                                exec(compile(''.join(cell['source']), str(path), 'exec'), namespace)


if __name__ == '__main__':
    unittest.main()
